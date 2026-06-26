##说明文档主要包含两方面介绍：解析工具使用方法 和 输出参数介绍

### 1. Adla Tool Use Instructions

adlainfo 主要用于解析xxx.adla模型文件，可以获取version版本号、节点nodes信息、buffer信息等，下面详细介绍下使用方法

        
#### 1.0 获取参数提示

	命令： ./adlainfo  --help
	结果：
		    ****************************************
			For example : ./adlainfo -v ***.adla
			Usage       : adlainfo [options] [adlafilepath]
			[Options]
				  -v                          : Get [v]ersion of adla compiler and board
				  -m                          : Get [m]emory size info
				  -io                         : Get [input/output] info
				  -H                          : Get [H]ardware info
				  -H -d                       : Get [H]ardware [d]etail info
				  -S                          : Get [S]oftware info
				  -S -d                       : Get [S]oftware [d]etail info
				  -H -S                       : Get [H]ardware and [S]oftware info
				  -H -S -d                    : Get [H]ardware and [S]oftware [d]etail info
				  -T                          : Get [T]ransformance adla info
				  --help                      : Get [help] to use Adla Tool
			****************************************

	解析：
			./adlainfo -v ***.adla			   获取***.adla 文件的版本，包括Aml Hw Platform："C3", "S5", "T7c", "T3x" 和compiler版本
			./adlainfo -m ***.adla			   获取***.adla 文件的内存信息
			./adlainfo -io ***.adla			   获取***.adla 文件的inputs/outputs信息
			./adlainfo -H ***.adla		       获取***.adla HardwareOp的信息
			./adlainfo -H -d ***.adla          获取***.adla HardwareOp的详细信息，可通过HardwareOp中的inputs/outputs 索引其详细的输入输出信息
			./adlainfo -S ***.adla		       获取***.adla SoftwareOp的信息
			./adlainfo -S -d ***.adla          获取***.adla SoftwareOp的详细信息，可通过SoftwareOp中的inputs/outputs 索引其详细的输入输出信息
			./adlainfo -H -S ***.adla		   获取***.adla HardwareOp/SoftwareOp的信息
			./adlainfo -H -S -d ***.adla       获取***.adla HardwareOp/SoftwareOp的详细信息，可通过HardwareOp/SoftwareOp中的inputs/outputs 索引其详细的输入输出信息
			./adlainfo -T ***.adla             获取***.adla eos_token, 设置的最大长度等信息
			./adlainfo --help				   获取输入参数提示
			./adlainfo -v				       获取adlainfo工具的版本

#### 1.1 获取版本信息

##### 1.1.1 获取adla文件的平台和compiler信息

	命令:  ./adlainfo -v mobilenet_v1_1.0_224_quant_u8.adla
	结果:
			****************************************
			Compile_Version       : 0.10.6
			Macc_count            : 568.99M
			Aml Hw Version        : 2.0
			****************************************
	解析: 
			Compile_Version : 生成adla文件的compiler版本信息
			Macc_count      : 统计计算的Macc单元
			Aml Hw Platform : 0.0 ———— C308L(AW419)/C302X(AW409)
							  1.0 ———— S928X
							  2.0 ———— AN400-A311D2
							  3.0 ———— BC302_T968D4
							  3.1 ———— BL201_S905X5

##### 1.1.2 获取adlainfo工具的版本

	命令:  ./adlainfo -v
	结果:
			ADLAInfo release version 1.7.1.3
	解析: 
			当前adlainfo的版本是1.7.1.3

#### 1.2 获取adla文件对应的内存信息
	命令:  ./adlainfo -m mobilenet_v1_1.0_224_quant_u8.adla 
	结果:
			****************************************
			[info]
				memory_size           : 4.21 M
				inputs_memory_size    : 147.00 kb
				outputs_memory_size   : 1001.00 bytes
			****************************************
	解析: 
			memory_size         : 内存使用
			inputs_memory_size  : int8/uint8 ———— input shape的乘积
								  int16 	 ———— input shape * 2
								  fp32		 ———— input shape * 4
			outputs_memory_size : int8/uint8 ———— 对应给定的buffer size，会大于等于output shape的乘积
								  int16 	 ———— 对应给定的buffer size，会大于等于output shape * 2的乘积
								  fp32		 ———— 对应给定的buffer size，会大于等于output shape * 4的乘积


##### 1.3 获取adla文件对应的Inputs/Outputs信息
	命令: ./adlainfo -io mobilenet_v1_1.0_224_quant_u8.adla
	结果:
			****************************************
				Inputs                  : [88 ]
					Input[0]            : 88
					Dim Count           : 4
					Size of dim[0]      : 1
					Size of dim[1]      : 224
					Size of dim[2]      : 224
					Size of dim[3]      : 3
					type                : Uint8
					scale               : [0.007812 ]
					zero_point          : [128 ]
					inputs_memory_size  : 147.00 kb
			****************************************

			****************************************
				Outputs                 : [87 ]
					Output[0]           : 87
					Dim Count           : 2
					Size of dim[0]      : 1
					Size of dim[1]      : 1001
					type                : Uint8
					scale               : [0.003906 ]
					zero_point          : [0 ]
					outputs_memory_size : 0.98 kb
			****************************************
				config:
				  max_outstanding_outputs : 4
				  axi_sram_size           : 1277952 bytes
			****************************************
	解析:
			Inputs           		: 表示adla输入/输出
			Input[0]         		: 表示adla输入的详细信息
			Dim Count         		: 输入的维度
			Size of dim       		: 输入的每个维度的值
			type              		: 输入的类型
			scale             		: 量化过程中的参数
			zero_point        		: 量化过程中的参数
			max_outstanding_outputs : 最大的输出维度
			axi_sram_size           : 设置axi_sram_size的size


#### 1.4 获取adla文件HardwareOp的信息
	命令: ./adlainfo -H mobilenet_v1_1.0_224_quant_u8.adla
	结果:
			[nodes] nodes_num : 17
			[nodes] Nodes list:
			****************************************
				node_index            : 0
				inputs                : [89 8 6 35 34 ]
				outputs               : [91 ]
				op                    : Hardware [Conv2d + DepthwiseConv2d]
				info:
					original_indices  : [0 1 ]
					original_operators: [Conv2d DepthwiseConv2d ]
			****************************************

			......

			****************************************
				node_index            : 15
				inputs                : [0 3 2 ]
				outputs               : [1 ]
				op                    : Hardware [Conv2d]
				info:
					original_indices  : [28 ]
					original_operators: [Conv2d ]
			****************************************

			Counts_HardwareOp         : 16
			****************************************
	解析: 
			node                  : 对应算子，分为hardware op和software op，compiler转换过程中可能把一个tflite中的op拆分成多个node
			nodes_num             : 统计node的数量
			node_index            : 表示Node编号
			inputs                : inputs是一个数组，存放的时node的input 对应tensor的index，outputs同理
			op                    : 表示Hardware/SoftwareOp
			Counts_HardwareOp     : 统计HardwareOp的个数
            original_indices      : [0 1 ]对应dump出原模型的type index
			original_operators    : [Conv2d DepthwiseConv2d ]获取对应dump出原模型ir node的type name
			Counts_HardwareOp     : 统计HardwareOp的个数


#### 1.5 获取adla文件HardwareOp的详细信息，可通过HardwareOp中的inputs/outputs 索引其详细的输入输出信息
	命令: ./adlainfo -H -d mobilenet_v1_1.0_224_quant_u8.adla
	结果:
			[nodes] nodes_num : 17
			[nodes] Nodes list:
			****************************************
				node_index            : 0
				inputs                : [89 8 6 35 34 ]
					inputs_index[0]   : 89
					buffer            : 47
					type              : Uint8
					shape             : [1 224 115 3 ]
					scale             : [0.007812 ]
					zero_point        : [128 ]

					inputs_index[1]   : 8
					buffer            : 66
					type              : Uint8
					shape             : [32 3 3 3 ]
					scale             : [0.021827 ]
					zero_point        : [151 ]

					inputs_index[2]   : 6
					buffer            : 23
					type              : Int16
					shape             : [1 1 1 32 ]
					scale             : [1.000000 ]
					zero_point        : [0 ]

					inputs_index[3]   : 35
					buffer            : 69
					type              : Uint8
					shape             : [1 3 3 32 ]
					scale             : [0.292199 ]
					zero_point        : [110 ]

					inputs_index[4]   : 34
					buffer            : 22
					type              : Int16
					shape             : [1 1 1 32 ]
					scale             : [1.000000 ]
					zero_point        : [0 ]

				outputs               : [91 ]
					outputs_index[0]  : 91
					buffer            : -1
					type              : Uint8
					shape             : [1 112 56 32 ]
					scale             : [0.023528 ]
					zero_point        : [0 ]

				op                    : Hardware [Conv2d + DepthwiseConv2d]
				info:
					original_indices  : [0 1 ]
					original_operators: [Conv2d DepthwiseConv2d ]
			****************************************
			
			......
			
			****************************************
				node_index            : 15
				inputs                : [0 3 2 ]
					inputs_index[0]   : 0
					buffer            : -1
					type              : Uint8
					shape             : [1 1 1 1024 ]
					scale             : [0.023528 ]
					zero_point        : [0 ]

					inputs_index[1]   : 3
					buffer            : 63
					type              : Uint8
					shape             : [1001 1 1 1024 ]
					scale             : [0.004987 ]
					zero_point        : [74 ]

					inputs_index[2]   : 2
					buffer            : 2
					type              : Int16
					shape             : [1 1 1 1001 ]
					scale             : [1.000000 ]
					zero_point        : [0 ]

				outputs               : [1 ]
					outputs_index[0]  : 1
					buffer            : 64
					type              : Uint8
					shape             : [1 1 1 1001 ]
					scale             : [0.166099 ]
					zero_point        : [66 ]

				op                    : Hardware [Conv2d]
				info:
					original_indices  : [28 ]
					original_operators: [Conv2d ]
			****************************************

			Counts_HardwareOp         : 16
			****************************************
	解析:
			nodes_num             : 统计node的数量
			node_index            : 表示Node编号
			inputs                : inputs是一个数组，存放的时node的input 对应tensor的index，outputs同理
			inputs_index   		  : 对应inputs数组中的数据，outputs_index同理对应outputs数组中的数据
			buffer                : 表明对应[buffers]中具体哪一个buffer;
			type                  : 存储数据类型
			shape                 : 切分后tensor的type
			scale                 : 量化过程中的参数
			zero_point            : 量化过程中的参数
			op                    : 表示Hardware/SoftwareOp
			original_indices      : [10 11 ]对应dump出原模型的type index
			original_operators    : [FullyConnected Add ]对应dump出原模型ir node的type name
			Counts_HardwareOp     : 统计HardwareOp的个数


#### 1.6 获取adla文件SoftwareOp的信息
	命令: ./adlainfo -S mobilenet_v1_1.0_224_quant_u8.adla
	结果:
			[nodes] nodes_num : 17
			[nodes] Nodes list:
			****************************************
				node_index            : 16
				inputs                : [4 ]
				outputs               : [87 ]
				op                    : Softmax
				info:
					original_indices  : [30 ]
					original_operators: [Softmax ]
			****************************************

			Counts_SoftwareOp         : 1
			****************************************
	解析: 
			node                  : 对应算子，分为hardware op和software op，compiler转换过程中可能把一个tflite中的op拆分成多个node
			nodes_num             : 统计node的数量
			node_index            : 表示Node编号
			inputs                : inputs是一个数组，存放的时node的input 对应tensor的index，outputs同理
			op                    : 表示Hardware/SoftwareOp
			original_indices      : [30 ]对应dump出原模型的type index
			original_operators    : [Softmax ]获取对应dump出原模型ir node的type name
			Counts_SoftwareOp     : 统计SoftwareOp的个数


#### 1.7 获取adla文件SoftwareOp的详细信息，可通过SoftwareOp中的inputs/outputs 索引其详细的输入输出信息
	命令: ./adlainfo -S -d mobilenet_v1_1.0_224_quant_u8.adla
	结果:
			[nodes] nodes_num : 17
			[nodes] Nodes list:
			****************************************
				node_index            : 16
				inputs                : [4 ]
					inputs_index[0]   : 4
					buffer            : 64
					type              : Uint8
					shape             : [1 1001 ]
					scale             : [0.166099 ]
					zero_point        : [66 ]

				outputs               : [87 ]
					outputs_index[0]  : 87
					buffer            : 65
					type              : Uint8
					shape             : [1 1001 ]
					scale             : [0.003906 ]
					zero_point        : [0 ]
				op                    : Softmax
				info:
					original_indices  : [30 ]
					original_operators: [Softmax ]
			****************************************

			Counts_SoftwareOp         : 1
			****************************************
	解析:
			nodes_num             : 统计node的数量
			node_index            : 表示Node编号
			inputs                : inputs是一个数组，存放的时node的input 对应tensor的index，outputs同理
			inputs_index   		  : 对应inputs数组中的数据，outputs_index同理对应outputs数组中的数据
			buffer                : 表明对应[buffers]中具体哪一个buffer;
			type                  : 存储数据类型
			shape                 : 切分后tensor的type
			scale                 : 量化过程中的参数
			zero_point            : 量化过程中的参数
			op                    : 表示Hardware/SoftwareOp
			original_indices      : [30 ]对应dump出原模型的type index
			original_operators    : [Softmax ]获取对应dump出原模型ir node的type name
			Counts_SoftwareOp     : 统计SoftwareOp的个数


#### 1.8 获取adla文件HardwareOp和SoftwareOp的信息
	命令: ./adlainfo -H -S mobilenet_v1_1.0_224_quant_u8.adla
	结果:
			[nodes] nodes_num : 17
			[nodes] Nodes list:
			****************************************
				node_index            : 0
				inputs                : [89 8 6 35 34 ]
				outputs               : [91 ]
				op                    : Hardware [Conv2d + DepthwiseConv2d]
				info:
					original_indices  : [0 1 ]
					original_operators: [Conv2d DepthwiseConv2d ]
			****************************************
			
			......
			
			****************************************
				node_index            : 16
				inputs                : [4 ]
				outputs               : [87 ]
				op                    : Softmax
				info:
					original_indices  : [30 ]
					original_operators: [Softmax ]
			****************************************

			Counts_HardwareOp         : 16
			Counts_SoftwareOp         : 1
			****************************************
	解析: 
			nodes_num             : 统计node的数量
			node_index            : 表示Node编号
			inputs                : inputs是一个数组，存放的时node的input 对应tensor的index，outputs同理
			op                    : 表示Hardware/SoftwareOp
			Counts_HardwareOp     : 统计HardwareOp的个数
			Counts_SoftwareOp     : 统计SoftwareOp的个数


#### 1.9 获取adla文件HardwareOp和SoftwareOp的详细信息，可通过HardwareOp和SoftwareOp中的inputs/outputs 索引其详细的输入输出信息
	命令: ./adlainfo -H -S -d mobilenet_v1_1.0_224_quant_u8.adla
	结果:
			[nodes] nodes_num : 17
			[nodes] Nodes list:
			****************************************
				node_index            : 0
				inputs                : [89 8 6 35 34 ]
					inputs_index[0]   : 89
					buffer            : 47
					type              : Uint8
					shape             : [1 224 115 3 ]
					scale             : [0.007812 ]
					zero_point        : [128 ]

					inputs_index[1]   : 8
					buffer            : 66
					type              : Uint8
					shape             : [32 3 3 3 ]
					scale             : [0.021827 ]
					zero_point        : [151 ]

					inputs_index[2]   : 6
					buffer            : 23
					type              : Int16
					shape             : [1 1 1 32 ]
					scale             : [1.000000 ]
					zero_point        : [0 ]

					inputs_index[3]   : 35
					buffer            : 69
					type              : Uint8
					shape             : [1 3 3 32 ]
					scale             : [0.292199 ]
					zero_point        : [110 ]

					inputs_index[4]   : 34
					buffer            : 22
					type              : Int16
					shape             : [1 1 1 32 ]
					scale             : [1.000000 ]
					zero_point        : [0 ]

				outputs               : [91 ]
					outputs_index[0]  : 91
					buffer            : -1
					type              : Uint8
					shape             : [1 112 56 32 ]
					scale             : [0.023528 ]
					zero_point        : [0 ]

				op                    : Hardware [Conv2d + DepthwiseConv2d]
				info:
					original_indices  : [0 1 ]
					original_operators: [Conv2d DepthwiseConv2d ]
			****************************************
			
			......
			
			****************************************
				node_index            : 16
				inputs                : [4 ]
					inputs_index[0]   : 4
					buffer            : 64
					type              : Uint8
					shape             : [1 1001 ]
					scale             : [0.166099 ]
					zero_point        : [66 ]

				outputs               : [87 ]
					outputs_index[0]  : 87
					buffer            : 65
					type              : Uint8
					shape             : [1 1001 ]
					scale             : [0.003906 ]
					zero_point        : [0 ]
				op                    : Softmax
				info:
					original_indices  : [30 ]
					original_operators: [Softmax ]
			****************************************

			Counts_HardwareOp         : 16
			Counts_SoftwareOp         : 1
			****************************************
	解析: 
			nodes_num             : 统计node的数量
			node_index            : 表示Node编号
			inputs                : inputs是一个数组，存放的时node的input 对应tensor的index，outputs同理
			inputs_index   		  : 对应inputs数组中的数据，outputs_index同理对应outputs数组中的数据
			buffer                : 表明对应[buffers]中具体哪一个buffer;
			type                  : 存储数据类型
			shape                 : 切分后tensor的type
			scale                 : 量化过程中的参数
			zero_point            : 量化过程中的参数
			op                    : 表示Hardware/SoftwareOp
			original_indices      : [30 ]对应dump出原模型的type index
			original_operators    : [Softmax ]获取对应dump出原模型ir node的type name
			Counts_HardwareOp     : 统计HardwareOp的个数
			Counts_SoftwareOp     : 统计SoftwareOp的个数

#### 1.10 获取transformer adla文件的配置信息
	命令: ./adlainfo_release -T Qwen2_5_05B_05_int16.adla
	结果:
			****************************************
			[TransformerConfig] list :
				vocab_size            : 151936
				max_sequence_length   : 512
			****************************************
	解析:
			vocab_size            : 词表大小
			max_sequence_length   : 模型支持的最大 token 序列长度
	
