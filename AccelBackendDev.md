# Verilator RTL 混合加速后端开发计划

## 项目概述

本项目计划开发一个基于 CPU 后端的混合加速方案，对于特定算子（如 Q8_0 量化矩阵乘法）使用 Verilator 编译的 RTL 仿真进行加速，其他算子继续使用原有的 CPU 实现。这是一个渐进式的加速方案，避免重新实现完整后端。

## 目标

**第一阶段目标：**
- 在 CPU 后端基础上集成 Verilator RTL 加速
- 实现 Q8_0 量化格式矩阵乘法的选择性加速
- 保持与现有 CPU 后端的完全兼容性
- 实现运行时动态选择（RTL 加速 vs CPU 实现）

## 技术背景

### Q8_0 量化格式
```c
#define QK8_0 32
typedef struct {
    ggml_half d;       // delta (scale factor)
    int8_t qs[QK8_0];  // quantized values
} block_q8_0;
```

Q8_0 是一种 8 位量化格式，每个块包含：
- 1 个 16 位浮点缩放因子
- 32 个 8 位量化值

### GGML CPU 后端架构
CPU 后端位于 `ggml/src/ggml-cpu/` 目录，包含：
- `ggml-cpu.c` - 主要实现文件，包含算子分发
- `ggml-cpu-impl.h` - 内部实现声明
- 各种架构特定的优化实现（x86、ARM等）

混合加速策略：
- 保持现有 CPU 后端不变
- 在关键算子中添加 Verilator 加速路径
- 通过条件编译和运行时检查选择执行路径

## 开发计划

### 阶段 1：项目设置和基础结构

#### 1.1 创建 Verilator 集成目录结构
```bash
mkdir -p ggml/src/ggml-cpu/verilator
mkdir -p ggml/src/ggml-cpu/rtl
mkdir -p tests/verilator
```

#### 1.2 创建基础文件
- `ggml/src/ggml-cpu/verilator/verilator-matmul.h` - Verilator 包装器声明
- `ggml/src/ggml-cpu/verilator/verilator-matmul.cpp` - Verilator 实现
- `ggml/src/ggml-cpu/ggml-cpu-verilator.h` - CPU 后端 Verilator 集成

#### 1.3 修改现有 CPU 后端
- 修改 `ggml-cpu.c` 中的算子分发逻辑
- 添加 Verilator 检测和初始化代码
- 保持向后兼容性

### 阶段 2：RTL 设计和 Verilator 集成

#### 2.1 设计 Q8_0 MatMul RTL 模块
```systemverilog
// q8_0_matmul.sv
module q8_0_matmul #(
    parameter ROWS = 1024,
    parameter COLS = 1024,
    parameter BLOCK_SIZE = 32
) (
    input  logic clk,
    input  logic rst_n,
    input  logic start,
    // Matrix A (Q8_0 format)
    input  logic [15:0] a_scale[],  // scale factors
    input  logic [7:0]  a_data[],   // quantized data
    // Matrix B (Q8_0 format) 
    input  logic [15:0] b_scale[],
    input  logic [7:0]  b_data[],
    // Output matrix
    output logic [31:0] result[],
    output logic done
);
```

#### 2.2 Verilator 集成
- 创建 Verilator 包装器类
- 实现 C++ 接口
- 处理内存映射和数据传输

```cpp
// verilator_wrapper.h
class VerilatorMatmul {
private:
    std::unique_ptr<Vq8_0_matmul> top;
    
public:
    VerilatorMatmul();
    ~VerilatorMatmul();
    
    void compute_matmul(
        const block_q8_0* a, int a_rows, int a_cols,
        const block_q8_0* b, int b_rows, int b_cols,
        float* result
    );
};
```

### 阶段 3：CPU 后端集成

#### 3.1 修改 CPU 后端算子分发

```cpp
// ggml/src/ggml-cpu/ggml-cpu.c

#ifdef GGML_USE_VERILATOR
#include "verilator/verilator-matmul.h"
static VerilatorMatmul* g_verilator_engine = nullptr;
static bool g_verilator_available = false;
#endif

// 在 ggml_compute_forward_mul_mat_q8_0 中添加 Verilator 路径
static void ggml_compute_forward_mul_mat_q8_0(
        const struct ggml_compute_params * params,
        struct ggml_tensor * dst) {

    const struct ggml_tensor * src0 = dst->src[0];
    const struct ggml_tensor * src1 = dst->src[1];

#ifdef GGML_USE_VERILATOR
    // 检查是否满足 Verilator 加速条件
    if (g_verilator_available && 
        should_use_verilator_matmul(src0, src1, dst)) {
        
        ggml_compute_forward_mul_mat_q8_0_verilator(params, dst);
        return;
    }
#endif

    // 原有 CPU 实现
    ggml_compute_forward_mul_mat_q8_0_cpu(params, dst);
}
```

#### 3.2 实现 Verilator 条件检查

```cpp
// ggml/src/ggml-cpu/ggml-cpu-verilator.h

#ifdef GGML_USE_VERILATOR

bool should_use_verilator_matmul(
    const struct ggml_tensor* src0,
    const struct ggml_tensor* src1, 
    const struct ggml_tensor* dst
) {
    // 检查矩阵大小是否适合 Verilator 加速
    const int64_t M = dst->ne[1];
    const int64_t N = dst->ne[0]; 
    const int64_t K = src0->ne[0];
    
    // 只对足够大的矩阵使用 Verilator
    if (M * N * K < VERILATOR_MIN_OPS_THRESHOLD) {
        return false;
    }
    
    // 检查数据类型
    if (src0->type != GGML_TYPE_Q8_0 || src1->type != GGML_TYPE_Q8_0) {
        return false;
    }
    
    // 检查内存布局
    if (!is_contiguous(src0) || !is_contiguous(src1)) {
        return false;
    }
    
    return true;
}

void ggml_compute_forward_mul_mat_q8_0_verilator(
    const struct ggml_compute_params * params,
    struct ggml_tensor * dst
);

#endif // GGML_USE_VERILATOR
```

#### 3.3 实现 Verilator 算子包装

```cpp
// ggml/src/ggml-cpu/verilator/verilator-matmul.cpp

void ggml_compute_forward_mul_mat_q8_0_verilator(
    const struct ggml_compute_params * params,
    struct ggml_tensor * dst
) {
    const struct ggml_tensor * src0 = dst->src[0];
    const struct ggml_tensor * src1 = dst->src[1];
    
    // 提取矩阵维度
    const int64_t ne00 = src0->ne[0]; // K
    const int64_t ne01 = src0->ne[1]; // M
    const int64_t ne10 = src1->ne[0]; // K  
    const int64_t ne11 = src1->ne[1]; // N
    
    // 获取数据指针
    const block_q8_0* a_data = (const block_q8_0*)src0->data;
    const block_q8_0* b_data = (const block_q8_0*)src1->data;
    float* result_data = (float*)dst->data;
    
    // 调用 Verilator 引擎
    if (g_verilator_engine) {
        g_verilator_engine->compute_matmul(
            a_data, ne01, ne00,  // M x K
            b_data, ne10, ne11,  // K x N  
            result_data          // M x N
        );
    } else {
        // 降级到 CPU 实现
        ggml_compute_forward_mul_mat_q8_0_cpu(params, dst);
    }
}
```

### 阶段 4：Verilator 引擎初始化

#### 4.1 实现 Verilator 引擎管理

``` cpp
// ggml/src/ggml-cpu/verilator/verilator-matmul.cpp

bool ggml_cpu_verilator_init() {
#ifdef GGML_USE_VERILATOR
    try {
        if (!g_verilator_engine) {
            g_verilator_engine = new VerilatorMatmul();
            g_verilator_available = true;
            GGML_LOG_INFO("Verilator RTL acceleration initialized\n");
        }
        return true;
    } catch (const std::exception& e) {
        GGML_LOG_ERROR("Failed to initialize Verilator: %s\n", e.what());
        g_verilator_available = false;
        return false;
    }
#else
    return false;
#endif
}

void ggml_cpu_verilator_free() {
#ifdef GGML_USE_VERILATOR
    if (g_verilator_engine) {
        delete g_verilator_engine;
        g_verilator_engine = nullptr;
        g_verilator_available = false;
        GGML_LOG_INFO("Verilator RTL acceleration freed\n");
    }
#endif
}
```


#### 4.2 集成到 CPU 后端初始化

```cpp
// ggml/src/ggml-cpu/ggml-cpu.c

// 在 CPU 后端初始化时调用
static void ggml_backend_cpu_init_once() {
    // 现有初始化代码...
    
#ifdef GGML_USE_VERILATOR
    // 尝试初始化 Verilator 加速
    ggml_cpu_verilator_init();
#endif
}

// 在 CPU 后端释放时调用
static void ggml_backend_cpu_free_resources() {
#ifdef GGML_USE_VERILATOR
    ggml_cpu_verilator_free();
#endif
}
```

### 阶段 5：构建系统集成

#### 5.1 修改 CPU 后端 CMakeLists.txt

```cmake
# ggml/src/ggml-cpu/CMakeLists.txt

# 添加 Verilator 支持选项
if (GGML_VERILATOR)
    find_package(verilator HINTS $ENV{VERILATOR_ROOT})
    if (verilator_FOUND)
        message(STATUS "Verilator found, enabling RTL acceleration for CPU backend")
        
        # Verilate RTL modules
        verilate(
            TARGET verilated_rtl
            SOURCES 
                ${CMAKE_CURRENT_SOURCE_DIR}/rtl/q8_0_matmul.sv
            VERILATOR_ARGS
                --cc --Wall --build -O3
        )
        
        # 添加 Verilator 源文件到 CPU 后端
        target_sources(ggml-cpu PRIVATE
            verilator/verilator-matmul.cpp
        )
        
        target_link_libraries(ggml-cpu PRIVATE verilated_rtl)
        target_compile_definitions(ggml-cpu PRIVATE GGML_USE_VERILATOR)
        
        target_include_directories(ggml-cpu PRIVATE
            ${CMAKE_CURRENT_SOURCE_DIR}/verilator
        )
        
    else()
        message(WARNING "Verilator not found, CPU backend will not have RTL acceleration")
    endif()
endif()
```

#### 5.2 修改主 CMakeLists.txt

```cmake
# 主 CMakeLists.txt 添加选项
option(GGML_VERILATOR "ggml: enable Verilator RTL acceleration for CPU backend" OFF)

# 传递给子目录
if (GGML_VERILATOR)
    set(GGML_VERILATOR ON CACHE BOOL "Enable Verilator" FORCE)
endif()
```

#### 5.3 条件编译配置

```cpp
// ggml/src/ggml-cpu/ggml-cpu-verilator.h

#ifndef GGML_CPU_VERILATOR_H
#define GGML_CPU_VERILATOR_H

#include "ggml.h"

#ifdef GGML_USE_VERILATOR

// Verilator 功能声明
bool ggml_cpu_verilator_init(void);
void ggml_cpu_verilator_free(void);
bool ggml_cpu_verilator_available(void);

bool should_use_verilator_matmul(
    const struct ggml_tensor* src0,
    const struct ggml_tensor* src1, 
    const struct ggml_tensor* dst
);

void ggml_compute_forward_mul_mat_q8_0_verilator(
    const struct ggml_compute_params * params,
    struct ggml_tensor * dst
);

#else

// 空实现
#define ggml_cpu_verilator_init() false
#define ggml_cpu_verilator_free() 
#define ggml_cpu_verilator_available() false
#define should_use_verilator_matmul(a,b,c) false

#endif // GGML_USE_VERILATOR

#endif // GGML_CPU_VERILATOR_H
```

### 阶段 6：测试和验证

#### 6.1 单元测试

```cpp
// tests/verilator/test-cpu-verilator.cpp
void test_cpu_verilator_q8_0_matmul() {
    // 测试 CPU 后端的 Verilator 加速功能
    const int M = 128, N = 256, K = 512;
    
    // 生成 Q8_0 格式的测试数据
    auto a_blocks = generate_q8_0_matrix(M, K);
    auto b_blocks = generate_q8_0_matrix(K, N);
    
    // 创建张量
    struct ggml_context* ctx = ggml_init({.mem_size = 1024*1024});
    struct ggml_tensor* a = ggml_new_tensor_2d(ctx, GGML_TYPE_Q8_0, K, M);
    struct ggml_tensor* b = ggml_new_tensor_2d(ctx, GGML_TYPE_Q8_0, K, N);
    struct ggml_tensor* result = ggml_mul_mat(ctx, a, b);
    
    // 设置数据
    memcpy(a->data, a_blocks.data(), a_blocks.size() * sizeof(block_q8_0));
    memcpy(b->data, b_blocks.data(), b_blocks.size() * sizeof(block_q8_0));
    
    // 计算图
    struct ggml_cgraph* graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, result);
    
    // 使用 CPU 后端计算（自动选择 Verilator 或 CPU）
    ggml_backend_t cpu_backend = ggml_backend_cpu_init();
    ggml_backend_graph_compute(cpu_backend, graph);
    
    // 验证结果正确性
    verify_q8_0_matmul_result(result, a_blocks, b_blocks);
    
    ggml_backend_free(cpu_backend);
    ggml_free(ctx);
}

void test_verilator_fallback() {
    // 测试当不满足条件时是否正确回退到 CPU
    // 例如：小矩阵、非连续内存等情况
}
```

#### 6.2 性能对比测试

```cpp
void benchmark_cpu_with_verilator() {
    const std::vector<int> sizes = {128, 256, 512, 1024, 2048};
    
    for (int size : sizes) {
        auto data = generate_q8_0_matrix(size, size);
        
        // 测试纯 CPU 实现（强制禁用 Verilator）
        auto cpu_time = benchmark_cpu_only_matmul(data);
        
        // 测试 CPU + Verilator 混合实现
        auto hybrid_time = benchmark_cpu_verilator_matmul(data);
        
        printf("Size %d: CPU=%.2fms, CPU+Verilator=%.2fms, Speedup=%.2fx\n",
               size, cpu_time, hybrid_time, cpu_time/hybrid_time);
    }
}
```

#### 6.3 运行时选择测试

```cpp
void test_runtime_selection() {
    // 测试不同条件下的算子选择
    struct test_case {
        int M, N, K;
        bool expect_verilator;
        const char* description;
    } cases[] = {
        {64, 64, 64, false, "Small matrix - should use CPU"},
        {1024, 1024, 1024, true, "Large matrix - should use Verilator"},
        {1, 4096, 4096, false, "Vector-matrix - may use CPU"},
    };
    
    for (auto& tc : cases) {
        bool used_verilator = test_matmul_selection(tc.M, tc.N, tc.K);
        GGML_ASSERT(used_verilator == tc.expect_verilator);
        printf("✓ %s\n", tc.description);
    }
}
```

### 阶段 7：集成测试和优化

#### 7.1 端到端测试
- 使用小型语言模型测试完整推理流程
- 验证 Q8_0 量化模型在混合后端下的正确性
- 测试不同算子组合的性能表现

#### 7.2 性能调优
- 优化 Verilator 触发条件（矩阵大小阈值）
- 调整数据传输和格式转换开销
- 实现算子融合优化

#### 7.3 稳定性测试
- 长时间运行测试
- 内存泄漏检测
- 多线程安全性验证

## 文件结构

```
llama.cpp/
├── AccelBackendDev.md                          # 本文档
├── ggml/
│   └── src/
│       └── ggml-cpu/
│           ├── ggml-cpu.c                      # 修改：添加 Verilator 分发逻辑
│           ├── ggml-cpu-verilator.h            # 新增：Verilator 集成接口
│           ├── verilator/
│           │   ├── verilator-matmul.h          # 新增：Verilator 包装器声明
│           │   ├── verilator-matmul.cpp        # 新增：Verilator 实现
│           │   └── CMakeLists.txt              # 修改：添加 Verilator 构建
│           └── rtl/
│               ├── q8_0_matmul.sv              # 新增：RTL 设计
│               ├── common.sv
│               └── testbench.sv
├── tests/
│   └── verilator/
│       ├── test-cpu-verilator.cpp              # 新增：混合后端测试
│       └── benchmark-cpu-verilator.cpp         # 新增：性能测试
└── cmake/
    └── FindVerilator.cmake                     # 新增：Verilator 查找脚本
```

## 开发时间估算

| 阶段 | 任务 | 预估时间 |
|------|------|----------|
| 1 | 项目设置和基础结构 | 1 天 |
| 2 | RTL 设计和 Verilator 集成 | 3-4 天 |
| 3 | CPU 后端集成和算子修改 | 3-4 天 |
| 4 | Verilator 引擎初始化 | 1-2 天 |
| 5 | 构建系统集成 | 1-2 天 |
| 6 | 测试和验证 | 2-3 天 |
| 7 | 集成测试和优化 | 2-3 天 |
| **总计** | | **13-19 天** |

## 技术挑战和解决方案

### 1. 算子选择逻辑
**挑战：** 如何智能地决定何时使用 Verilator 加速
**解决方案：** 
- 基于矩阵大小的阈值判断
- 考虑数据布局和内存连续性
- 提供环境变量控制选择策略

### 2. 性能平衡
**挑战：** Verilator 仿真可能不总是比优化的 CPU 代码快
**解决方案：** 
- 实现智能的性能监控和自适应选择
- 针对特定大小范围优化
- 提供性能分析工具

### 3. 兼容性维护
**挑战：** 保持与现有 CPU 后端的完全兼容性
**解决方案：** 
- 最小化对现有代码的修改
- 通过条件编译确保可选性
- 完整的回退机制

### 4. 调试复杂性
**挑战：** RTL 和软件混合调试的复杂性
**解决方案：** 
- 详细的日志和诊断信息
- 独立的 RTL 测试平台
- 渐进式集成和验证

## 扩展计划

### 短期扩展
1. 支持其他量化格式（Q4_0, Q5_0）的 Verilator 加速
2. 实现多算子 Verilator 加速（卷积、激活函数等）
3. 添加自适应性能阈值调整
4. 支持异步 Verilator 计算

### 长期扩展
1. 扩展到其他后端（GPU 后端的 RTL 加速）
2. 实现真实硬件后端（FPGA、ASIC）
3. 支持模型并行和流水线加速
4. 开发专用的量化算子库

## 验收标准

1. **功能正确性：** 通过所有单元测试，Verilator 加速结果与纯 CPU 实现匹配
2. **性能指标：** 在特定大小的矩阵上实现可测量的性能提升
3. **兼容性：** 完全向后兼容，不影响现有 CPU 后端功能
4. **可选性：** 可通过构建选项禁用，不依赖 Verilator 时正常工作
5. **智能选择：** 能够根据条件智能选择 Verilator 或 CPU 实现
6. **代码质量：** 遵循项目编码标准，包含充分的文档和测试

## 下一步行动

1. **立即开始：** 创建基础项目结构和 Verilator 集成目录
2. **第一周：** 完成 RTL 设计和 Verilator 包装器
3. **第二周：** 修改 CPU 后端，集成 Verilator 算子路径
4. **第三周：** 完成构建系统集成和测试验证

这个修改后的开发计划提供了一个更实用的混合加速方案，在保持完全兼容性的同时，为特定算子提供 Verilator RTL 加速能力。
