import sys
import triton
import triton.language as tl
from triton_anchor.compiler import unified_compile
from triton_anchor.plugins.base import BackendPlugin
from triton_anchor.hw_capability import HWCapability, ComputeParadigm

# ═════════════════════════════════════════════════════════════════════════════
# 1. 定义一个简单的 Triton Kernel
# ═════════════════════════════════════════════════════════════════════════════
@triton.jit
def add_kernel(x_ptr, y_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    output = x + y
    tl.store(output_ptr + offsets, output, mask=mask)

# ═════════════════════════════════════════════════════════════════════════════
# 2. 定义一个 Dummy Backend Plugin，模拟硬件后端
# ═════════════════════════════════════════════════════════════════════════════
class DummyPlugin(BackendPlugin):
    @property
    def name(self) -> str:
        return "dummy_backend"
    
    @property
    def hw_capability(self) -> HWCapability:
        from triton_anchor.hw_capability import TensorCapability
        return HWCapability(
            name="dummy_backend",
            arch_family="tpu",
            compute_paradigm=ComputeParadigm.TENSOR_PROCESSOR,
            anchor_ir_track="linalg",
            ptr_model="axis_info",
            tensor_cap=TensorCapability()
        )
    
    def lower_anchor_ir_to_target(self, anchor_ir, metadata: dict) -> bytes:
        # 这里应该调用 LLVM 或者硬件专有编译器
        # 我们在这里打印转换成功的 AnchorIR
        print("\n[DummyPlugin] 收到 AnchorIR:")
        print("--------------------------------------------------")
        print(str(anchor_ir))
        print("--------------------------------------------------")
        return b"dummy_binary_output"
        
    def create_launcher(self, binary: bytes, metadata: dict) -> object:
        return lambda *args, **kwargs: None

    def validate_environment(self) -> bool:
        return True

# ═════════════════════════════════════════════════════════════════════════════
# 3. 提供 AST 编译需要的 Dummy Options (因为我们没有真正激活目标后端)
# ═════════════════════════════════════════════════════════════════════════════
class DummyOptions:
    def __init__(self):
        self.num_warps = 4
        self.num_stages = 3
        self.num_ctas = 1
        self.cluster_dims = (1, 1, 1)
        self.ptx_version = None
        self.enable_fp_fusion = True
        self.supported_fp8_dtypes = ()
        self.deprecated_fp8_dtypes = ()
        self.allowed_dot_input_precisions = ("ieee", "tf32", "tf32x3")
        self.allow_fp8e4nv = False
        self.max_num_imprecise_acc_default = False
        self.debug = False

def main():
    print(f"📦 Triton Version: {triton.__version__}")
    print(f"📦 Python Version: {sys.version.split()[0]}")
    
    print("[1] 初始化 MLIR Context...")
    from triton._C.libtriton import ir
    ctx = ir.context()
    
    print("[2] 加载 Dialects (Triton + Anchor)...")
    ir.load_dialects(ctx)
    from triton._C.libtriton import anchor
    anchor.load_dialects(ctx)
    
    print("[3] 将 Python 函数编译为 TTIR...")
    # 构建 ASTSource
    src = triton.compiler.ASTSource(
        fn=add_kernel,
        signature={0: "*fp32", 1: "*fp32", 2: "*fp32", 3: "i32"},
        constants={4: 256},
    )
    
    # 转换为初始的 TTIR
    options = DummyOptions()
    ttir_module = src.make_ir(options=options, codegen_fns=None, context=ctx)
    
    print("\n[4] 调用 triton-anchor 的统一编译管线 (unified_compile)...")
    plugin = DummyPlugin()
    
    # 执行 TTIR -> AnchorIR (Linalg) -> Dummy Backend
    metadata = {"name": "add_kernel"}
    # 关闭 anchor_ir 严格校验，因为测试环境可能未注册全部扩展 dialects
    result = unified_compile(ttir_module, plugin, metadata, validate_anchor_ir=False)
    
    print(f"\n✅ 编译成功！生成二进制大小: {len(result['binary'])} bytes")

if __name__ == "__main__":
    main()
