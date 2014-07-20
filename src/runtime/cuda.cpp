#include "mini_stdint.h"
#include "../buffer_t.h"
#include "HalideRuntime.h"
#include "mini_cuda.h"
#include "cuda_opencl_shared.h"

#ifdef DEBUG
#define DEBUG_PRINTF halide_printf
#else
// This ensures that DEBUG and non-DEBUG have the same semicolon eating behavior.
static void _noop_printf(void *, const char *, ...) { }
#define DEBUG_PRINTF _noop_printf
#endif

extern "C" {

extern int atoi(const char *);
extern char *getenv(const char *);
extern int64_t halide_current_time_ns(void *user_context);
extern void *malloc(size_t);
extern int snprintf(char *, size_t, const char *, ...);

// A cuda context defined in this module with weak linkage
CUcontext WEAK weak_cuda_ctx = 0;
volatile int WEAK weak_cuda_lock = 0;

// A pointer to the cuda context to use, which may not be the one above. This pointer is followed at init_kernels time.
CUcontext WEAK *cuda_ctx_ptr = NULL;
volatile int WEAK *cuda_lock_ptr = NULL;

WEAK void halide_set_cuda_context(CUcontext *ctx_ptr, volatile int *lock_ptr) {
    cuda_ctx_ptr = ctx_ptr;
    cuda_lock_ptr = lock_ptr;
}

static CUresult create_context(void *user_context, CUcontext *ctx);

// The default implementation of halide_acquire_cl_context uses the global
// pointers above, and serializes access with a spin lock.
// Overriding implementations of acquire/release must implement the following
// behavior:
// - halide_acquire_cl_context should always store a valid context/command
//   queue in ctx/q, or return an error code.
// - A call to halide_acquire_cl_context is followed by a matching call to
//   halide_release_cl_context. halide_acquire_cl_context should block while a
//   previous call (if any) has not yet been released via halide_release_cl_context.
WEAK int halide_acquire_cuda_context(void *user_context, CUcontext *ctx) {
    // TODO: Should we use a more "assertive" assert? these asserts do
    // not block execution on failure.
    halide_assert(user_context, ctx != NULL);

    if (cuda_ctx_ptr == NULL) {
        cuda_ctx_ptr = &weak_cuda_ctx;
        cuda_lock_ptr = &weak_cuda_lock;
    }

    halide_assert(user_context, cuda_lock_ptr != NULL);
    while (__sync_lock_test_and_set(cuda_lock_ptr, 1)) { }

    // If the context has not been initialized, initialize it now.
    halide_assert(user_context, cuda_ctx_ptr != NULL);
    if (*cuda_ctx_ptr == NULL) {
        CUresult error = create_context(user_context, cuda_ctx_ptr);
        if (error != CUDA_SUCCESS) {
            __sync_lock_release(cuda_lock_ptr);
            return error;
        }
    }

    *ctx = *cuda_ctx_ptr;
    return 0;
}

WEAK int halide_release_cuda_context(void *user_context) {
    __sync_lock_release(cuda_lock_ptr);
    return 0;
}

}

// Helper object to acquire and release the OpenCL context.
class CudaContext {
    void *user_context;

public:
    CUcontext context;
    int error;

    // Constructor sets 'error' if any occurs.
    CudaContext(void *user_context) : user_context(user_context),
                                      context(NULL),
                                      error(CUDA_SUCCESS) {
        error = halide_acquire_cuda_context(user_context, &context);
        halide_assert(user_context, context != NULL);
        if (error != 0) {
            return;
        }

        error = cuCtxPushCurrent(context);
    }

    ~CudaContext() {
        CUcontext old;
        cuCtxPopCurrent(&old);

        halide_release_cuda_context(user_context);
    }
};

extern "C" {
// Structure to hold the state of a module attached to the context.
// Also used as a linked-list to keep track of all the different
// modules that are attached to a context in order to release them all
// when then context is released.
struct _module_state_ WEAK *state_list = NULL;
typedef struct _module_state_ {
    CUmodule module;
    _module_state_ *next;
} module_state;

WEAK bool halide_validate_dev_pointer(void *user_context, buffer_t* buf, size_t size=0) {
// The technique using cuPointerGetAttribute and CU_POINTER_ATTRIBUTE_CONTEXT
// requires unified virtual addressing is enabled and that is not the case
// for 32-bit processes on Mac OS X. So for now, as a total hack, just return true
// in 32-bit. This could of course be wrong the other way for cards that only
// support 32-bit addressing in 64-bit processes, but I expect those cards do not
// support unified addressing at all.
// TODO: figure out a way to validate pointers in all cases if strictly necessary.
#ifdef BITS_32
    return true;
#else
    if (buf->dev == 0)
        return true;

    CUcontext ctx;
    CUresult result = cuPointerGetAttribute(&ctx, CU_POINTER_ATTRIBUTE_CONTEXT, buf->dev);
    if (result) {
        halide_printf(user_context, "Bad device pointer %p: cuPointerGetAttribute returned %d\n",
                      (void *)buf->dev, result);
        return false;
    }
    return true;
#endif
}

WEAK int halide_dev_free(void *user_context, buffer_t* buf) {
    // halide_dev_free, at present, can be exposed to clients and they
    // should be allowed to call halide_dev_free on any buffer_t
    // including ones that have never been used with a GPU.
    if (buf->dev == 0) {
        return 0;
    }

    DEBUG_PRINTF( user_context, "CUDA: halide_dev_free (user_context: %p, buf: %p)\n", user_context, buf );

    CudaContext ctx(user_context);
    if (ctx.error != CUDA_SUCCESS)
        return ctx.error;

    #ifdef DEBUG
    uint64_t t_before = halide_current_time_ns(user_context);
    #endif

    halide_assert(user_context, halide_validate_dev_pointer(user_context, buf));

    DEBUG_PRINTF( user_context, "    cuMemFree %p\n", buf->dev );
    CUresult err = cuMemFree(buf->dev);
    // If cuMemFree fails, it isn't likely to succeed later, so just drop
    // the reference.
    buf->dev = 0;
    if (err != CUDA_SUCCESS) {
        halide_error_varargs(user_context, "CUDA: cuMemFree failed (%d)", err);
        return err;
    }

    #ifdef DEBUG
    uint64_t t_after = halide_current_time_ns(user_context);
    halide_printf(user_context, "    Time: %f ms\n", (t_after - t_before) / 1.0e6);
    #endif

    return 0;
}

static CUresult create_context(void *user_context, CUcontext *ctx) {
    // Initialize CUDA
    CUresult err = cuInit(0);
    if (err != CUDA_SUCCESS) {
        halide_error_varargs(user_context, "CUDA: cuInit failed (%d)", err);
        return err;
    }

    // Make sure we have a device
    int deviceCount = 0;
    err = cuDeviceGetCount(&deviceCount);
    if (err != CUDA_SUCCESS) {
        halide_error_varargs(user_context, "CUDA: cuGetDeviceCount failed (%d)", err);
        return err;
    }
    if (deviceCount <= 0) {
        halide_error(user_context, "CUDA: No devices available");
        return CUDA_ERROR_NO_DEVICE;
    }

    int device = halide_get_gpu_device(user_context);
    if (device == -1) {
        device = deviceCount - 1;
    }

    // Get device
    CUdevice dev;
    CUresult status = cuDeviceGet(&dev, device);
    if (status != CUDA_SUCCESS) {
        halide_error(user_context, "CUDA: Failed to get device\n");
        return status;
    }

    DEBUG_PRINTF( user_context, "    Got device %d\n", dev );

    // Dump device attributes
    #ifdef DEBUG
    {
        char name[256];
        name[0] = 0;
        err = cuDeviceGetName(name, 256, dev);
        DEBUG_PRINTF(user_context, "      %s\n", name);

        if (err != CUDA_SUCCESS) {
            halide_error_varargs(user_context, "CUDA: cuDeviceGetName failed (%d)", err);
            return err;
        }

        size_t memory = 0;
        err = cuDeviceTotalMem(&memory, dev);
        DEBUG_PRINTF(user_context, "      total memory: %d MB\n", (int)(memory >> 20));

        if (err != CUDA_SUCCESS) {
            halide_error_varargs(user_context, "CUDA: cuDeviceTotalMem failed (%d)", err);
            return err;
        }

        // Declare variables for other state we want to query.
        int max_threads_per_block = 0, warp_size = 0, num_cores = 0;
        int max_block_size[] = {0, 0, 0};
        int max_grid_size[] = {0, 0, 0};
        int max_shared_mem = 0, max_constant_mem = 0;
        int cc_major = 0, cc_minor = 0;

        struct {int *dst; CUdevice_attribute attr;} attrs[] = {
            {&max_threads_per_block, CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK},
            {&warp_size,             CU_DEVICE_ATTRIBUTE_WARP_SIZE},
            {&num_cores,             CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT},
            {&max_block_size[0],     CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_X},
            {&max_block_size[1],     CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Y},
            {&max_block_size[2],     CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Z},
            {&max_grid_size[0],      CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_X},
            {&max_grid_size[1],      CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Y},
            {&max_grid_size[2],      CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Z},
            {&max_shared_mem,        CU_DEVICE_ATTRIBUTE_SHARED_MEMORY_PER_BLOCK},
            {&max_constant_mem,      CU_DEVICE_ATTRIBUTE_TOTAL_CONSTANT_MEMORY},
            {&cc_major,              CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR},
            {&cc_minor,              CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR},
            {NULL,                   CU_DEVICE_ATTRIBUTE_MAX}};

        // Do all the queries.
        for (int i = 0; attrs[i].dst; i++) {
            err = cuDeviceGetAttribute(attrs[i].dst, attrs[i].attr, dev);
            if (err != CUDA_SUCCESS) {
                halide_error_varargs(user_context,
                                     "CUDA: cuDeviceGetAttribute failed (%d) for attribute %d",
                                     err, (int)attrs[i].attr);
                return err;
            }
        }

        // threads per core is a function of the compute capability
        int threads_per_core = (cc_major == 1 ? 8 :
                                cc_major == 2 ? (cc_minor == 0 ? 32 : 48) :
                                cc_major == 3 ? 192 :
                                cc_major == 5 ? 128 : 0);

        DEBUG_PRINTF(user_context,
                     "      max threads per block: %d\n"
                     "      warp size: %d\n"
                     "      max block size: %d %d %d\n"
                     "      max grid size: %d %d %d\n"
                     "      max shared memory per block: %d\n"
                     "      max constant memory per block: %d\n"
                     "      compute capability %d.%d\n"
                     "      cuda cores: %d x %d = %d\n",
                     max_threads_per_block, warp_size,
                     max_block_size[0], max_block_size[1], max_block_size[2],
                     max_grid_size[0], max_grid_size[1], max_grid_size[2],
                     max_shared_mem, max_constant_mem,
                     cc_major, cc_minor,
                     num_cores, threads_per_core, num_cores * threads_per_core);
    }
    #endif

    // Create context
    DEBUG_PRINTF( user_context, "    cuCtxCreate %d -> ", dev );
    err = cuCtxCreate(ctx, 0, dev);
    if (err != CUDA_SUCCESS) {
        DEBUG_PRINTF( user_context, "%d\n", err );
        halide_error_varargs(user_context, "CUDA: cuCtxCreate failed (%d)", err);
        return err;
    } else {
        unsigned int version = 0;
        cuCtxGetApiVersion(*ctx, &version);
        DEBUG_PRINTF( user_context, "%p (%d)\n", *ctx, version);
    }

    return CUDA_SUCCESS;
}

WEAK int halide_init_kernels(void *user_context, void **state_ptr, const char* ptx_src, int size) {
    DEBUG_PRINTF( user_context, "CUDA: halide_init_kernels (user_context: %p, state_ptr: %p, ptx_src: %p, %i)\n",
                  user_context, state_ptr, ptx_src, size );

    CudaContext ctx(user_context);
    if (ctx.error != 0) {
        return ctx.error;
    }

    #ifdef DEBUG
    uint64_t t_before = halide_current_time_ns(user_context);
    #endif

    // Create the state object if necessary. This only happens once, regardless
    // of how many times halide_init_kernels/halide_release is called.
    // halide_release traverses this list and releases the module objects, but
    // it does not modify the list nodes created/inserted here.
    module_state **state = (module_state**)state_ptr;
    if (!(*state)) {
        *state = (module_state*)malloc(sizeof(module_state));
        (*state)->module = NULL;
        (*state)->next = state_list;
        state_list = *state;
    }

    // Create the module itself if necessary.
    if (!(*state)->module) {
        DEBUG_PRINTF( user_context, "    cuModuleLoadData %p, %i -> ", ptx_src, size );
        CUmodule module;
        CUresult err = cuModuleLoadData(&(*state)->module, ptx_src);
        if (err != CUDA_SUCCESS) {
            DEBUG_PRINTF( user_context, "%d\n", err );
            halide_error_varargs(user_context, "CUDA: cuModuleLoadData failed (%d)", err);
            return err;
        } else {
            DEBUG_PRINTF( user_context, "%p\n", module );
        }
    }

    #ifdef DEBUG
    uint64_t t_after = halide_current_time_ns(user_context);
    halide_printf(user_context, "    Time: %f ms\n", (t_after - t_before) / 1.0e6);
    #endif

    return 0;
}

WEAK void halide_release(void *user_context) {
    DEBUG_PRINTF( user_context, "CUDA: halide_release (user_context: %p)\n", user_context );

    int err;
    CUcontext ctx;
    err = halide_acquire_cuda_context(user_context, &ctx);
    if (err != CUDA_SUCCESS || !ctx) {
        return;
    }

    // It's possible that this is being called from the destructor of
    // a static variable, in which case the driver may already be
    // shutting down.
    err = cuCtxSynchronize();
    halide_assert(user_context, err == CUDA_SUCCESS || err == CUDA_ERROR_DEINITIALIZED);

    // Unload the modules attached to this context. Note that the list
    // nodes themselves are not freed, only the module objects are
    // released. Subsequent calls to halide_init_kernels might re-create
    // the program object using the same list node to store the module
    // object.
    module_state *state = state_list;
    while (state) {
        if (state->module) {
            DEBUG_PRINTF(user_context, "    cuModuleUnload %p\n", state->module);
            err = cuModuleUnload(state->module);
            halide_assert(user_context, err == CUDA_SUCCESS || err == CUDA_ERROR_DEINITIALIZED);
            state->module = 0;
        }
        state = state->next;
    }

    // Only destroy the context if we own it
    if (ctx == weak_cuda_ctx) {
        DEBUG_PRINTF(user_context, "    cuCtxDestroy %p\n", weak_cuda_ctx);
        err = cuCtxDestroy(weak_cuda_ctx);
        halide_assert(user_context, err == CUDA_SUCCESS || err == CUDA_ERROR_DEINITIALIZED);
        weak_cuda_ctx = NULL;
    }

    halide_release_cuda_context(user_context);
}

WEAK int halide_dev_malloc(void *user_context, buffer_t *buf) {
    DEBUG_PRINTF( user_context, "CUDA: halide_dev_malloc (user_context: %p, buf: %p)\n", user_context, buf );

    CudaContext ctx(user_context);
    if (ctx.error != CUDA_SUCCESS) {
        return ctx.error;
    }

    size_t size = _buf_size(user_context, buf);
    if (buf->dev) {
        // This buffer already has a device allocation
        halide_assert(user_context, halide_validate_dev_pointer(user_context, buf, size));
        return 0;
    }

    halide_assert(user_context, buf->stride[0] >= 0 && buf->stride[1] >= 0 &&
                                buf->stride[2] >= 0 && buf->stride[3] >= 0);

    DEBUG_PRINTF(user_context, "    allocating buffer of %lld bytes, "
                 "extents: %lldx%lldx%lldx%lld strides: %lldx%lldx%lldx%lld (%d bytes per element)\n",
                 (long long)size,
                 (long long)buf->extent[0], (long long)buf->extent[1],
                 (long long)buf->extent[2], (long long)buf->extent[3],
                 (long long)buf->stride[0], (long long)buf->stride[1],
                 (long long)buf->stride[2], (long long)buf->stride[3],
                 buf->elem_size);

    #ifdef DEBUG
    uint64_t t_before = halide_current_time_ns(user_context);
    #endif

    CUdeviceptr p;
    DEBUG_PRINTF( user_context, "    cuMemAlloc %lld -> ", size );
    CUresult err = cuMemAlloc(&p, size);
    if (err != CUDA_SUCCESS) {
        DEBUG_PRINTF( user_context, "%d\n", err );
        halide_error_varargs(user_context, "CUDA: cuMemAlloc failed (%d)", err);
        return err;
    } else {
        DEBUG_PRINTF( user_context, "%p\n", p );
    }
    halide_assert(user_context, p);
    buf->dev = (uint64_t)p;

    #ifdef DEBUG
    uint64_t t_after = halide_current_time_ns(user_context);
    halide_printf(user_context, "    Time: %f ms\n", (t_after - t_before) / 1.0e6);
    #endif

    return 0;
}

WEAK int halide_copy_to_dev(void *user_context, buffer_t* buf) {
    int err = halide_dev_malloc(user_context, buf);
    if (err) {
        return err;
    }

    DEBUG_PRINTF( user_context, "CUDA: halide_copy_to_dev (user_context: %p, buf: %p)\n", user_context, buf );

    CudaContext ctx(user_context);
    if (ctx.error != CUDA_SUCCESS) {
        return ctx.error;
    }

    if (buf->host_dirty) {
        #ifdef DEBUG
        uint64_t t_before = halide_current_time_ns(user_context);
        #endif

        halide_assert(user_context, buf->host && buf->dev);
        halide_assert(user_context, halide_validate_dev_pointer(user_context, buf));

        _dev_copy c = _make_host_to_dev_copy(buf);

        for (int w = 0; w < c.extent[3]; w++) {
            for (int z = 0; z < c.extent[2]; z++) {
                for (int y = 0; y < c.extent[1]; y++) {
                    for (int x = 0; x < c.extent[0]; x++) {
                        uint64_t off = (x * c.stride_bytes[0] +
                                        y * c.stride_bytes[1] +
                                        z * c.stride_bytes[2] +
                                        w * c.stride_bytes[3]);
                        void *src = (void *)(c.src + off);
                        CUdeviceptr dst = (CUdeviceptr)(c.dst + off);
                        uint64_t size = c.chunk_size;
                        DEBUG_PRINTF( user_context, "    cuMemcpyHtoD (%d, %d, %d, %d), %p -> %p, %lld bytes\n",
                                      x, y, z, w,
                                      src, (void *)dst, (long long)size );
                        CUresult err = cuMemcpyHtoD(dst, src, size);
                        if (err != CUDA_SUCCESS) {
                            halide_error_varargs(user_context, "CUDA: cuMemcpyHtoD failed (%d)", err);
                            return err;
                        }
                    }
                }
            }
        }


        #ifdef DEBUG
        uint64_t t_after = halide_current_time_ns(user_context);
        halide_printf(user_context, "    Time: %f ms\n", (t_after - t_before) / 1.0e6);
        #endif
    }
    buf->host_dirty = false;
    return 0;
}

WEAK int halide_copy_to_host(void *user_context, buffer_t* buf) {
    if (!buf->dev_dirty) {
        return 0;
    }

    DEBUG_PRINTF( user_context, "CUDA: halide_copy_to_host (user_context: %p, buf: %p)\n", user_context, buf );

    CudaContext ctx(user_context);
    if (ctx.error != CUDA_SUCCESS) {
        return ctx.error;
    }

    // Need to check dev_dirty again, in case another thread did the
    // copy_to_host before the serialization point above.
    if (buf->dev_dirty) {
        #ifdef DEBUG
        uint64_t t_before = halide_current_time_ns(user_context);
        #endif

        halide_assert(user_context, buf->dev && buf->dev);
        halide_assert(user_context, halide_validate_dev_pointer(user_context, buf));

        _dev_copy c = _make_dev_to_host_copy(buf);

        for (int w = 0; w < c.extent[3]; w++) {
            for (int z = 0; z < c.extent[2]; z++) {
                for (int y = 0; y < c.extent[1]; y++) {
                    for (int x = 0; x < c.extent[0]; x++) {
                        uint64_t off = (x * c.stride_bytes[0] +
                                        y * c.stride_bytes[1] +
                                        z * c.stride_bytes[2] +
                                        w * c.stride_bytes[3]);
                        CUdeviceptr src = (CUdeviceptr)(c.src + off);
                        void *dst = (void *)(c.dst + off);
                        uint64_t size = c.chunk_size;
                        DEBUG_PRINTF( user_context, "    cuMemcpyDtoH (%d, %d, %d, %d), %p -> %p, %lld bytes\n",
                                      x, y, z, w,
                                      (void *)src, dst, (long long)size );
                        CUresult err = cuMemcpyDtoH(dst, src, size);
                        if (err != CUDA_SUCCESS) {
                            halide_error_varargs(user_context, "CUDA: cuMemcpyDtoH failed (%d)", err);
                            return err;
                        }
                    }
                }
            }
        }

        #ifdef DEBUG
        uint64_t t_after = halide_current_time_ns(user_context);
        halide_printf(user_context, "    Time: %f ms\n", (t_after - t_before) / 1.0e6);
        #endif
    }
    buf->dev_dirty = false;
    return 0;
}

// Used to generate correct timings when tracing
WEAK int halide_dev_sync(void *user_context) {
    DEBUG_PRINTF( user_context, "CUDA: halide_dev_sync (user_context: %p)\n", user_context );

    CudaContext ctx(user_context);
    if (ctx.error != CUDA_SUCCESS) {
        return ctx.error;
    }

    #ifdef DEBUG
    uint64_t t_before = halide_current_time_ns(user_context);
    #endif

    CUresult err = cuCtxSynchronize();
    if (err != CUDA_SUCCESS) {
        halide_error_varargs(user_context, "CUDA: cuCtxSynchronize failed (%d)", err);
        return err;
    }

    #ifdef DEBUG
    uint64_t t_after = halide_current_time_ns(user_context);
    halide_printf(user_context, "    Time: %f ms\n", (t_after - t_before) / 1.0e6);
    #endif

    return 0;
}

WEAK int halide_dev_run(void *user_context,
                        void *state_ptr,
                        const char* entry_name,
                        int blocksX, int blocksY, int blocksZ,
                        int threadsX, int threadsY, int threadsZ,
                        int shared_mem_bytes,
                        size_t arg_sizes[],
                        void* args[]) {
    DEBUG_PRINTF( user_context, "CUDA: halide_dev_run (user_context: %p, entry: %s, blocks: %dx%dx%d, threads: %dx%dx%d, shmem: %d)\n",
                  user_context, entry_name,
                  blocksX, blocksY, blocksZ,
                  threadsX, threadsY, threadsZ,
                  shared_mem_bytes );

    CUresult err;
    CudaContext ctx(user_context);
    if (ctx.error != CUDA_SUCCESS) {
        return ctx.error;
    }

    #ifdef DEBUG
    uint64_t t_before = halide_current_time_ns(user_context);
    #endif

    halide_assert(user_context, state_ptr);
    CUmodule mod = ((module_state*)state_ptr)->module;
    halide_assert(user_context, mod);
    CUfunction f;
    err = cuModuleGetFunction(&f, mod, entry_name);
    if (err != CUDA_SUCCESS) {
        halide_error_varargs(user_context, "CUDA: cuModuleGetFunction failed (%d)", err);
        return err;
    }

    err = cuLaunchKernel(f,
                         blocksX,  blocksY,  blocksZ,
                         threadsX, threadsY, threadsZ,
                         shared_mem_bytes,
                         NULL, // stream
                         args,
                         NULL);
    if (err != CUDA_SUCCESS) {
        halide_error_varargs(user_context, "CUDA: cuLaunchKernel failed (%d)", err);
        return err;
    }

    #ifdef DEBUG
    err = cuCtxSynchronize();
    if (err != CUDA_SUCCESS) {
        halide_error_varargs(user_context, "CUDA: cuCtxSynchronize failed (%d)\n", err);
        return err;
    }
    uint64_t t_after = halide_current_time_ns(user_context);
    halide_printf(user_context, "    Time: %f ms\n", (t_after - t_before) / 1.0e6);
    #endif
    return 0;
}

} // extern "C" linkage
