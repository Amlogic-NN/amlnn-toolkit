1. Introduction
     In response to the problem that Google runtime does not support some operators, which leads to model splitting and thus affects operating efficiency, a set of operator reconstruction tools is developed to refactor the operators that introduce splitting and optimize the pass in the nnapi processing flow to improve the model operating efficiency

2. Function introduction
     This tool is currently developed based on our NNAPI, so it will be reconstructed according to the read model. Referring to tensorflow, we will split and merge some operators. The current model conversion Operator improvement during the process will show which operators have been modified in the model conversion result
     The current version will have the following operator improvements:
        1. For the int8 type of operator in tflite, but NNAPI does not support the int8 type of the operator
        Operators involved: abs, exp, log, rsqrt, sqrt, sin, l2norm, pow
        Note: If there is no optimization and refactoring, the model will be split into multiple graphs

3. How to use
    3.1 Get parameter hints
    Commands: ./tflite_refactorer --help
    Resluts:

    Guide:
    usage
        ./tflite_refactorer original_name.tflite new_name.tflite
    
    Parse:
        original_name.tflite is the original model name
        new_name.tflite is the name of the reconstructed model

    3.2 Model reconstruction
    Commands:./tflite_refactorer dped_instance_quant.tflite test.tflite
    Results:

    operation RSQRT index 8 modify from int8 to float32
    operation RSQRT index 19 modify from int8 to float32
    operation RSQRT index 31 modify from int8 to float32
    operation RSQRT index 42 modify from int8 to float32
    operation RSQRT index 54 modify from int8 to float32
    operation RSQRT index 65 modify from int8 to float32
    operation RSQRT index 77 modify from int8 to float32
    operation RSQRT index 88 modify from int8 to float32
	
	
    ExportOperators
    IExportTensors
    ExportInputTensors
    ExportOutputTensors
    Generate refactorer tflite: dped_instance_quant_modify.tflite

    Parse:
        1.If there is a similar: index keyword, it means that there is a scene that needs to be refactored, and the refactoring will take effect. When integrating, please use the refactored tflite model. In other cases, it is recommended to use the original model.
        2.The number after index is the position information of the operator that needs to be modified in the original model. See the location information of the operator after the model is opened with the netron tool.

4.Common problem
    4.1 Model parsing error, tool exits normally
        The error message is as follows:
		[Error] this is the model that all operations load as ADD.  This kind of model is not support temporarily
		[Note] exit the model refactor. Suggest provide this model to Amlogic NN engineer

		suggestion:
            Record and feed back the original model to NN engineers

    4.2 Segmentation problem occurs in the tool
        The error message is as follows:
        Segmentation fault (core dumped)
	
		suggestion:
        1. You can get more logs by setting the environment variable export NNRT_LOG_LEVEL=3 and executing the tool again, and feedback the error log to the Amlogic engineer
        2. If the log cannot support the problem location, it is recommended to provide the original model to Amlogic  NN engineer
