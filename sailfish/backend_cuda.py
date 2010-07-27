import operator

import pycuda.autoinit
import pycuda.compiler
import pycuda.tools
import pycuda.driver as cuda

from struct import calcsize, pack

def _expand_block(block):
    if block is int:
        return (block, 1, 1)
    elif len(block) == 1:
        return (block[0], 1, 1)
    elif len(block) == 2:
        return (block[0], block[1], 1)
    else:
        return block

def _expand_grid(grid):
    if len(grid) == 1:
        return (grid[0], 1)
    else:
        return grid

def _set_txt_format(dsc, strides):
    # float
    if strides[-1] == 4:
        dsc.format = cuda.array_format.FLOAT
        dsc.num_channels = 1
    # double encoded as int2
    else:
        dsc.format = cuda.array_format.UNSIGNED_INT32
        dsc.num_channels = 2

class CUDABackend(object):

    @classmethod
    def add_options(cls, group):
        group.add_option('--cuda-kernel-stats', dest='cuda_kernel_stats',
                help='print information about amount of memory and registers used by the kernels', action='store_true', default=False)
        group.add_option('--cuda-nvcc-opts', dest='cuda_nvcc_opts',
                help='additional parameters to pass to the CUDA compiler', action='store', type='string', default='')
        group.add_option('--cuda-keep-temp', dest='cuda_keep_temp',
                help='keep intermediate CUDA files', action='store_true', default=False)
        return 1

    def __init__(self, options):
        self.buffers = {}
        self._kern_stats = set()
        self._tex_to_memcpy = {}
        self.options = options

    def alloc_buf(self, size=None, like=None):
        if like is not None:
            # When calculating the total array size, take into account
            # any striding.
            buf = cuda.mem_alloc(like.shape[0] * like.strides[0])

            if like.base is not None:
                self.buffers[buf] = like.base
            else:
                self.buffers[buf] = like

            self.to_buf(buf)
        else:
            buf = cuda.mem_alloc(size)

        return buf

    def nonlocal_field(self, prog, cl_buf, num, shape, strides):
        if len(shape) == 3:
            dsc = cuda.ArrayDescriptor()
            dsc.width = strides[0] / strides[2]
            dsc.height = shape[-3]
            _set_txt_format(dsc, strides)

            txt = prog.get_texref('img_f%d' % num)
            txt.set_address_2d(cl_buf, dsc, strides[-3])

            # It turns out that using 3D textures doesn't really make
            # much sense if it requires copying data around.  We therefore
            # access the 3D fields via a 2D texture, which still provides
            # some caching, while not requiring a separate copy of the
            # data.
            #
            # dsc = cuda.ArrayDescriptor3D()
            # dsc.depth, dsc.height, dsc.width = shape
            # dsc.format = cuda.array_format.FLOAT
            # dsc.num_channels = 1
            # ary = cuda.Array(dsc)

            # copy = cuda.Memcpy3D()
            # copy.set_src_device(cl_buf)
            # copy.set_dst_array(ary)
            # copy.width_in_bytes = copy.src_pitch = strides[-2]
            # copy.src_height = copy.height = dsc.height
            # copy.depth = dsc.depth

            # txt = prog.get_texref('img_f%d' % num)
            # txt.set_array(ary)
            # self._tex_to_memcpy[txt] = copy
        else:
            # 2D texture.
            dsc = cuda.ArrayDescriptor()
            dsc.width = shape[-1]
            dsc.height = shape[-2]
            _set_txt_format(dsc, strides)
            txt = prog.get_texref('img_f%d' % num)
            txt.set_address_2d(cl_buf, dsc, strides[-2])
        return txt

    def to_buf(self, cl_buf, source=None):
        if source is None:
            if cl_buf in self.buffers:
                cuda.memcpy_htod(cl_buf, self.buffers[cl_buf])
            else:
                raise ValueError('Unknown compute buffer and source not specified.')
        else:
            if source.base is not None:
                cuda.memcpy_htod(cl_buf, source.base)
            else:
                cuda.memcpy_htod(cl_buf, source)

    def from_buf(self, cl_buf, target=None):
        if target is None:
            if cl_buf in self.buffers:
                cuda.memcpy_dtoh(self.buffers[cl_buf], cl_buf)
            else:
                raise ValueError('Unknown compute buffer and target not specified.')
        else:
            if target.base is not None:
                cuda.memcpy_dtoh(target.base, cl_buf)
            else:
                cuda.memcpy_dtoh(target, cl_buf)

    def build(self, source):
        if self.options.cuda_nvcc_opts:
            import shlex
            options = shlex.split(self.options.cuda_nvcc_opts)
        else:
            options = []

        return pycuda.compiler.SourceModule(source, options=options, keep=self.options.cuda_keep_temp) #options=['-Xopencc', '-O0']) #, options=['--use_fast_math'])

    def get_kernel(self, prog, name, block, args, args_format, shared=None, fields=[]):
        """FIXME

        :param args: can be None
        """
        kern = prog.get_function(name)
        kern.param_set_size(calcsize(args_format))
        setattr(kern, 'args', [args, args_format])
        setattr(kern, 'img_fields', [x for x in fields if x is not None])
        kern.set_block_shape(*_expand_block(block))
        if shared is not None:
            kern.set_shared_size(shared)

        if self.options.cuda_kernel_stats and name not in self._kern_stats:
            self._kern_stats.add(name)
            ddata = pycuda.tools.DeviceData()
            occ = pycuda.tools.OccupancyRecord(ddata, reduce(operator.mul, block), kern.shared_size_bytes, kern.num_regs)

            print '%s: l:%d  s:%d  r:%d  occ:(%f tb:%d w:%d l:%s)' % (name, kern.local_size_bytes, kern.shared_size_bytes,
                    kern.num_regs, occ.occupancy, occ.tb_per_mp, occ.warps_per_mp, occ.limited_by)

        return kern

    def get_args(self, kern):
        """Return a list of arguments set for the kernel."""
        return kern.args[0]

    def run_kernel(self, kernel, grid_size, args=None):
        """Run a CUDA kernel.

        :param kernel: kernel object obtained from :func:`get_kernel`
        :param grid_size: size of the grid (tuple)
        :param args: if not None, arguments with which to call the kernel
        """
        if args is not None:
            kernel.param_setv(0, pack(kernel.args[1], *args))
        else:
            kernel.param_setv(0, pack(kernel.args[1], *kernel.args[0]))
        for img_field in kernel.img_fields:
            # Copy device buffer to 3D CUDA array if neessary.
            # if img_field in self._tex_to_memcpy:
            #    self._tex_to_memcpy[img_field]()
            kernel.param_set_texref(img_field)
        kernel.launch_grid(*_expand_grid(grid_size))

    def sync(self):
        cuda.Context.synchronize()

    def get_defines(self):
        return {
            'shared_var': '__shared__',
            'kernel': '__global__',
            'global_ptr': '',
            'const_ptr': '',
            'device_func': '__device__',
            'const_var': '__constant__',
        }


backend=CUDABackend
