##The documentation contains two main aspects: how to use the parsing tool and an introduction to the output parameters

### 1. Adla Tool Use Instructions

adlainfo Mainly used to parse ***.adla model file, you can get version version number, node nodes information, buffer information, etc., the following details of the use

        
#### 1.0 Get parameter tips

	Command: ./adlainfo  --help
	Results:
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

	Parse:
			./adlainfo -v ***.adla			   Get the version of ***.adla file, including Aml Hw Platform: "C308L(AW419)/C302X(AW409)", "S928X", "AN400-A311D2", "BC302_T968D4" and compiler version
			./adlainfo -m ***.adla			   Get the memory information of ***.adla file
			./adlainfo -io ***.adla			   Get inputs/outputs information of ***.adla file
			./adlainfo -H ***.adla		       Get information about ***.adla HardwareOp
			./adlainfo -H -d ***.adla          Get detailed information about ***.adla HardwareOp, which can be indexed by inputs/outputs in HardwareOp
			./adlainfo -S ***.adla		       Get information about ***.adla SoftwareOp
			./adlainfo -S -d ***.adla          Get detailed information about ***.adla SoftwareOp, which can be indexed by inputs/outputs in SoftwareOp
			./adlainfo -H -S ***.adla		   Get information about ***.adla HardwareOp/SoftwareOp
			./adlainfo -H -S -d ***.adla       Get the details of ***.adla HardwareOp/SoftwareOp. Detailed input and output information can be indexed via inputs/outputs in HardwareOp/SoftwareOp
			./adlainfo -T ***.adla             Get ***.adla eos_token, set max length, etc.
			./adlainfo --help				   Get input parameter tips
			./adlainfo -v				       Get version of adlainfo tool

#### 1.1 Get version information

##### 1.1.1 Get the platform and compiler information of the adla file

	Command:  ./adlainfo -v mobilenet_v1_1.0_224_quant_u8.adla 
	Results:
			****************************************
			Compile_Version       : 0.10.6
			Macc_count            : 568.99M
			Aml Hw Version        : 2.0
			****************************************
	Parse: 
			Compile_Version : Generate compiler version information for adla files
			Macc_count      : Macc unit for statistical calculations
			Aml Hw Platform : 0.0 ———— C308L(AW419)/C302X(AW409)
							  1.0 ———— S928X
							  2.0 ———— AN400-A311D2
							  3.0 ———— BC302_T968D4
							  3.1 ———— BL201_S905X5

##### 1.1.2 Get version of adlainfo tool

	Command:  ./adlainfo -v
	Results:
			ADLAInfo release version 1.7.1.3
	Parse: 
			Current version of adlainfo is 1.7.1.3

#### 1.2 Get the memory information corresponding to the adla file
	Command:  ./adlainfo -m mobilenet_v1_1.0_224_quant_u8.adla
	Results:
			****************************************
			[info]
				memory_size           : 4.21 M
				inputs_memory_size    : 147.00 kb
				outputs_memory_size   : 1001.00 bytes
			****************************************
	Parse: 
			memory_size         : Memory Usage
			inputs_memory_size  : int8/uint8 ———— Product of input shape
								  int16 	 ———— input shape * 2
								  fp32		 ———— input shape * 4
			outputs_memory_size : int8/uint8 ———— Corresponding to a given buffer size, it will be greater than or equal to the product of output shape
								  int16 	 ———— Corresponding to a given buffer size, it will be greater than or equal to the product of output shape * 2
								  fp32		 ———— Corresponding to a given buffer size, it will be greater than or equal to the product of output shape * 4


##### 1.3 Get the Inputs/Outputs information corresponding to the adla file
	Command: ./adlainfo -io mobilenet_v1_1.0_224_quant_u8.adla
	Results:
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
	Parse:
			Inputs           		: Indicates adla input/output
			Input[0]         		: Indicates adla input details
			Dim Count         		: Dimension of input
			Size of dim       		: The value of each dimension of the input
			type              		: Type of input
			scale             		: Parameters in the quantification process
			zero_point        		: Parameters in the quantification process
			max_outstanding_outputs : Maximum output dimension
			axi_sram_size           : Set the size of axi_sram_size


#### 1.4 Get information about the adla file HardwareOp
	Command: ./adlainfo -H mobilenet_v1_1.0_224_quant_u8.adla
	Results:
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
	Parse: 
			node                  : Corresponding operator, divided into hardware op and software op, the model conversion process may split the op in an original model into multiple nodes
			nodes_num             : Counting the number of nodes
			node_index            : Indicates the Node number
			inputs                : Inputs is an array that holds the inputs of the node corresponding to the index of the tensor, and the outputs are the same
			op                    : Indicates Hardware/SoftwareOp
			original_indices      : [10 11 ]Dump the type index of the original model
			original_operators    : [FullyConnected Add ]Get the type name of the original model ir node corresponding to the dump
			Counts_HardwareOp     : Count the number of HardwareOp


#### 1.5 Get detailed information about ***.adla HardwareOp, which can be indexed by inputs/outputs in HardwareOp
	Command: ./adlainfo -H -d mobilenet_v1_1.0_224_quant_u8.adla
	Results:
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
	Parse:
			nodes_num             : Counting the number of nodes
			node_index            : Indicates the Node number
			inputs                : Inputs is an array that holds the inputs of the node corresponding to the index of the tensor, and the outputs are the same
			inputs_index   		  : Corresponds to the data in the inputs array, and outputs_index corresponds to the data in the outputs array
			buffer                : Indicate which buffer in [buffers] corresponds to the specific
			type                  : Storage Data Type
			shape                 : Type of tensor after slicing
			scale                 : Parameters in the quantification process
			zero_point            : Parameters in the quantification process
			op                    : Hardware/SoftwareOp
			original_indices      : [10 11 ]Dump the type index of the original model
			original_operators    : [FullyConnected Add ]Get the type name of the original model ir node corresponding to the dump
			Counts_HardwareOp     : Count the number of HardwareOp


#### 1.6 Get information about the adla file SoftwareOp
	Command: ./adlainfo -S mobilenet_v1_1.0_224_quant_u8.adla
	Results:
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
	Parse: 
			node                  : Corresponding operator, divided into hardware op and software op, the model conversion process may split the op in an original model into multiple nodes
			nodes_num             : Counting the number of nodes
			node_index            : Indicates the Node number
			inputs                : Inputs is an array that holds the inputs of the node corresponding to the index of the tensor, and the outputs are the same
			op                    : Hardware/SoftwareOp
			original_indices      : [30 ]Dump the type index of the original model
			original_operators    : [Softmax ]Get the type name of the original model ir node corresponding to the dump
			Counts_SoftwareOp     : Count the number of SoftwareOp


#### 1.7 Get detailed information about ***.adla SoftwareOp, which can be indexed by inputs/outputs in SoftwareOp
	Command: ./adlainfo -S -d mobilenet_v1_1.0_224_quant_u8.adla
	Results:
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
	Parse:
			nodes_num             : Counting the number of nodes
			node_index            : Indicates the Node number
			inputs                : Inputs is an array that holds the inputs of the node corresponding to the index of the tensor, and the outputs are the same
			inputs_index   		  : Corresponds to the data in the inputs array, and outputs_index corresponds to the data in the outputs array
			buffer                : Indicate which buffer in [buffers] corresponds to the specific
			type                  : Storage Data Type
			shape                 : Type of tensor after slicing
			scale                 : Parameters in the quantification process
			zero_point            : Parameters in the quantification process
			op                    : Hardware/SoftwareOp
			original_indices      : [30 ]Dump the type index of the original model
			original_operators    : [Softmax ]Get the type name of the original model ir node corresponding to the dump
			Counts_SoftwareOp     : Count the number of SoftwareOp


#### 1.8 Get information about adla file HardwareOp and SoftwareOp
	Command: ./adlainfo -H -S mobilenet_v1_1.0_224_quant_u8.adla
	Results:
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
	Parse: 
			nodes_num             : Counting the number of nodes
			node_index            : Indicates the Node number
			inputs                : Inputs is an array that holds the inputs of the node corresponding to the index of the tensor, and the outputs are the same
			op                    : Hardware/SoftwareOp
			original_indices      : [30 ]Dump the type index of the original model
			original_operators    : [Softmax ]Get the type name of the original model ir node corresponding to the dump
			Counts_HardwareOp     : Count the number of HardwareOp
			Counts_SoftwareOp     : Count the number of SoftwareOp


#### 1.9 Get detailed information about the adla files HardwareOp and SoftwareOp, you can use the inputs/outputs in HardwareOp and SoftwareOp to index their detailed input and output information.
	Command: ./adlainfo -H -S -d mobilenet_v1_1.0_224_quant_u8.adla
	Results:
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
	Parse: 
			nodes_num             : Counting the number of nodes
			node_index            : Indicates the Node number
			inputs                : Inputs is an array that holds the inputs of the node corresponding to the index of the tensor, and the outputs are the same
			inputs_index   		  : Corresponds to the data in the inputs array, and outputs_index corresponds to the data in the outputs array
			buffer                : Indicate which buffer in [buffers] corresponds to the specific
			type                  : Storage Data Type
			shape                 : Type of tensor after slicing
			scale                 : Parameters in the quantification process
			zero_point            : Parameters in the quantification process
			op                    : Hardware/SoftwareOp
			original_indices      : [30 ]Dump the type index of the original model
			original_operators    : [Softmax ]Get the type name of the original model ir node corresponding to the dump
			Counts_HardwareOp     : Count the number of HardwareOp
			Counts_SoftwareOp     : Count the number of SoftwareOp

#### 1.10 Get configuration information for the transformer adla file
	Command: ./adlainfo_release -T Qwen2_5_05B_05_int16.adla
	Results:
			****************************************
			[TransformerConfig] list :
				vocab_size            : 151936
				max_sequence_length   : 512
			****************************************
	Parse:
			vocab_size            : vocabulary list size
			max_sequence_length   : Max token sequence length supported by the model