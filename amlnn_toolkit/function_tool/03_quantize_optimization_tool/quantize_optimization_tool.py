#
# Copyright (C) 2024–2025 Amlogic, Inc. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

import sys
import os
from scipy import spatial
import numpy as np
import tensorflow as tf
import amlnn
from amlnn.api import AMLNN
amlnn =  AMLNN(log_level='DEBUG')



class aml_nn_model:
    def __init__(self):

        self.mdoel_type='pt2'
        self.model_path='/xxxxx/resnet50_exported.pt2'
        self.source_file='/xxxxxx/ILSVRC2012_val_256.txt'
        self.normalization_mean = [[123.675, 116.28, 103.53]]
        self.normalization_std =  [[58.395, 57.12, 57.375]]
        self.itreation=32
        self.outdir='./'
        self.input_shape=[224,224,3]
        self.input_npy_path=['/xxxx/resnet50_nhwc.npy']
        self.output_npy_path=['/xxxx/resnet50_out_0.npy']
        self.activation_quant_algo=["normal",'percentile',"histogram","omse"]
        self.weight_quant_algo =["normal","lsq"]
        self.inference_input_type="float32"
        self.inference_output_type="float32"
        self.percentile_sigma =[0.02,0.04,0.06,0.08,0.1,0.12,0.14,0.16,0.18,0.2]
        self.percentile_alpha =[0.9999,0.99999,0.999999,0.9999999]
        self.omse_range =[80,85,90,95,98]
        self.omse_step =[0.01,0.012,0.014,0.016,0.018,0.02]
        self.num_histogram_bins = [2048, 4096, 8192 , 12288,16384]
        os.environ['ADLA_ENABLE_CHANNEL_LAST_IO']=str(True)
        self.result="test_result.txt"
        self.output=[]
        self.euclidean_dis=[]
        self.cos_sim=[]

    def run_tflite(self):
        print("Load Model:{}".format(self.middle_tflite))
        interpreter = tf.lite.Interpreter(model_path=self.middle_tflite,num_threads=8)
        interpreter.allocate_tensors()

        for i in range(len(interpreter.get_input_details())):
            index = interpreter.get_input_details()[i]['index']
            name = interpreter.get_input_details()[i]["name"]
            shape = interpreter.get_input_details()[i]["shape"]
            dtype = interpreter.get_input_details()[i]["dtype"]
            quantization = interpreter.get_input_details()[i]["quantization"]
            print("\nInput[{}]: index:{},name:{} shape:{}, dtype:{}, quantization:{}".format(i, index, name, shape, dtype, quantization))

            data = np.load(self.input_npy_path[i])
            interpreter.set_tensor(index, data)

        interpreter.invoke()

        for i in range(len(interpreter.get_output_details())):
            index = interpreter.get_output_details()[i]['index']
            name = interpreter.get_output_details()[i]["name"]
            shape = interpreter.get_output_details()[i]["shape"]
            dtype = interpreter.get_output_details()[i]["dtype"]
            quantization = interpreter.get_output_details()[i]["quantization"]
            print("\nOutput[{}]: index:{}, shape:{}, dtype:{}, quantization:{}".format(i, index, name, shape, dtype, quantization))
            out = interpreter.get_tensor(index)
            self.output.append(out)
        os.system("rm -rf {} ".format(self.middle_tflite))

    def compute_euclidean_dis_and_cos(self):
        if len(self.output) != len(self.output_npy_path):
            print("Error: The number of outputs does not match the number of outputs provided")
            exit()

        for i in range(len(self.output)):
            temp1=self.output[i]
            temp2=np.load(self.output_npy_path[i])

            squart_dis = tf.square(tf.subtract(temp1,temp2))
            dis_sum = tf.reduce_sum(squart_dis)
            euclidean = tf.sqrt(dis_sum)

            #cos_sim_temp = 1 - spatial.distance.cosine(temp1.reshape(1, -1), temp2.reshape(1, -1))
            cos_sim_temp = 1 - spatial.distance.cosine(temp1.reshape(-1), temp2.reshape(-1))
            self.euclidean_dis.append(float(euclidean.numpy()))
            self.cos_sim.append(cos_sim_temp)
            print("\n Result cos_sim:{},euclidean_dis:{}\n".format(self.cos_sim[-1], self.euclidean_dis[-1]))

    def record_result(self,cmd_line):
        with open(self.result, 'a') as record:
             record.write(cmd_line+" \n")
             for i in range(len(self.output)):
                 record.write("output[{}]:cosine similarity:{:-<8f},Euclidean distance:{:-<8f} \n".format(i, self.cos_sim[i], self.euclidean_dis[i]))
             record.write("+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ \n")
        self.euclidean_dis=[]
        self.cos_sim=[]
        self.output=[]

    def test_result(self,cmd_line):
        self.run_tflite()

        self.compute_euclidean_dis_and_cos()

        self.record_result(cmd_line)

def main(argv):
    aml_model=aml_nn_model()
    if aml_model.mdoel_type=="pt2":
        amlnn.load_pt2(aml_model.model_path)
    elif aml_model.model_type=="tflite":
        amlnn.load_tflite(aml_model.model_path)
    elif aml_model.mdoel_type=="pt":
        amlnn.load_pytorch(aml_model.model_path)
    elif aml_model.model_type=="onnx":
        amlnn.load_onnx(aml_model.model_path)
    for activation_quant_algo in aml_model.activation_quant_algo:
        if activation_quant_algo=="percentile":
            for percentile_sigma in aml_model.percentile_sigma:
                for percentile_alpha in aml_model.percentile_alpha:
                    for weight_quant_algo in aml_model.weight_quant_algo:
                        amlnn.config(normalization_mean=aml_model.normalization_mean,
                                     normalization_std=aml_model.normalization_std, 
                                     quantized_dtype='w8a8',
                                     activation_quant_algo=activation_quant_algo,
                                     weight_quant_algo=weight_quant_algo,
                                     target_platform='PRODUCT_PID0XA003', 
                                     export_intermediate=True,
                                     output_dir=aml_model.outdir,
                                     percentile_sigma=percentile_sigma,
                                     percentile_alpha=percentile_alpha,
                                     inference_input_type=aml_model.inference_input_type,
                                     inference_output_type=aml_model.inference_output_type)
                        amlnn.compile(dataset=aml_model.source_file, iterations=aml_model.itreation)
                        aml_model.middle_tflite=amlnn.base.params_config["model_ir_path"]
                        cmd_line=" -- activation_quant_algo {} --percentile_sigma {} --percentile_alpha {} --weight_quant_algo {} ".format(
                        activation_quant_algo,percentile_sigma,percentile_alpha,weight_quant_algo)
                        aml_model.test_result(cmd_line)
        elif activation_quant_algo=="omse":
            for omse_range in aml_model.omse_range:
                for omse_steps in aml_model.omse_step:
                    for weight_quant_algo in aml_model.weight_quant_algo:
                        amlnn.config(normalization_mean=aml_model.normalization_mean,
                                     normalization_std=aml_model.normalization_std, 
                                     quantized_dtype='w8a8',
                                     activation_quant_algo=activation_quant_algo, 
                                     weight_quant_algo=weight_quant_algo,
                                     target_platform='PRODUCT_PID0XA003', 
                                     export_intermediate=True,
                                     output_dir=aml_model.outdir,
                                     omse_range=omse_range,
                                     omse_steps=omse_steps,
                                     inference_input_type=aml_model.inference_input_type,
                                     inference_output_type=aml_model.inference_output_type)
                        amlnn.compile(dataset=aml_model.source_file, iterations=aml_model.itreation)
                        aml_model.middle_tflite=amlnn.base.params_config["model_ir_path"]
                        cmd_line=" -- activation_quant_algo {} --omse_steps {} --omse_range {} --weight_quant_algo {} ".format(
                        activation_quant_algo,omse_steps,omse_range,weight_quant_algo)
                        aml_model.test_result(cmd_line)
        elif activation_quant_algo=="histogram":
            for histogram_bins in aml_model.num_histogram_bins:
                    for weight_quant_algo in aml_model.weight_quant_algo:
                        amlnn.config(normalization_mean=aml_model.normalization_mean,
                                     normalization_std=aml_model.normalization_std, 
                                     quantized_dtype='w8a8',
                                     activation_quant_algo=activation_quant_algo, 
                                     weight_quant_algo=weight_quant_algo,
                                     target_platform='PRODUCT_PID0XA003', 
                                     export_intermediate=True,
                                     output_dir=aml_model.outdir,
                                     histogram_bins=histogram_bins,
                                     inference_input_type=aml_model.inference_input_type,
                                     inference_output_type=aml_model.inference_output_type)
                        amlnn.compile(dataset=aml_model.source_file, iterations=aml_model.itreation)
                        aml_model.middle_tflite=amlnn.base.params_config["model_ir_path"]
                        cmd_line=" -- activation_quant_algo {} --histogram_bins {} --weight_quant_algo {} ".format(
                        activation_quant_algo,histogram_bins,weight_quant_algo)
                        aml_model.test_result(cmd_line)
        elif activation_quant_algo=="normal":
            for weight_quant_algo in aml_model.weight_quant_algo:
                amlnn.config(normalization_mean=aml_model.normalization_mean,
                             normalization_std=aml_model.normalization_std, 
                             quantized_dtype='w8a8',
                             activation_quant_algo=activation_quant_algo, 
                             weight_quant_algo=weight_quant_algo,
                             target_platform='PRODUCT_PID0XA003', 
                             export_intermediate=True,
                             output_dir=aml_model.outdir,
                             inference_input_type=aml_model.inference_input_type,
                             inference_output_type=aml_model.inference_output_type)
                amlnn.compile(dataset=aml_model.source_file, iterations=aml_model.itreation)
                aml_model.middle_tflite=amlnn.base.params_config["model_ir_path"]
                cmd_line=" -- activation_quant_algo {} --weight_quant_algo {} ".format(
                activation_quant_algo,weight_quant_algo)
                aml_model.test_result(cmd_line)            
        



if __name__ == '__main__':
    main(sys.argv)
