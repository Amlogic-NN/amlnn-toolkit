一.简介
    针对Google runtime对于部分算子不支持导致模型拆分从而影响运行效率的问题，开发了一套算子重构工具对于引入拆分的算子进行重构并在nnapi处理流程进行pass优化以提升模型运行效率

二.功能介绍
    本工具目前是基于我们的NNAPI进行开发的,故目前会根据读取到的模型进行重构，参照tensorflow的具体验证，我们会对部分算子进行算子拆分和算子合并,目前模型转换过程中进行算子改进的情况会在模型转换结果中显示哪些算子进行了修改
    当前版本会有以下算子改进情况:
		1、针对tflite中算子存在int8类型，但是NNAPI不支持该算子int8类型的情况
			涉及算子：abs、exp、log、rsqrt、sqrt、sin、l2norm、pow
			Note：如果没有优化重构，会出现模型被拆分成多个graph情况
    

三.使用方法
    3.1 获取参数提示
    命令: ./tflite_refactorer --help
    结果:
    Guide:
    usage
        ./tflite_refactorer original_name.tflite new_name.tflite
    解析:
        original_name.tflite为原始模型名称
        new_name.tflite为重构模型的名称

    3.2 模型重构
    执行命令:./tflite_refactorer dped_instance_quant.tflite dped_instance_quant_modify.tflite
    结果:
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
    解析：
		1.如果出现类似：index 关键字，表示当前出现需要重构的场景，重构生效，集成时，请用重构后的tflite模型。其他情况，建议使用原始模型即可.
        2.index 后面的数字为原始模型中出现需要修改的算子的位置信息,见模型通过netron工具打开后算子的location信息

四.常见问题
    4.1  模型解析错误，工具正常退出问题
		报错信息如下：
		[Error] this is the model that all operations load as ADD.  This kind of model is not support temporarily
		[Note] exit the model refactor. Suggest provide this model to NN engineer

		建议：
			将原始模型记录并反馈给Amlogic NN工程师

    4.2 工具出现Segmentation问题
		报错信息如下：
		Segmentation fault (core dumped)
	
		建议：
			1、可以通过设置环境变量export NNRT_LOG_LEVEL=3并再次执行工具获取更多的log，将错误log反馈给Amlogic工程师
			2、如果log不能支撑问题定位，建议将原始模型提供给Amlogic NN工程师

