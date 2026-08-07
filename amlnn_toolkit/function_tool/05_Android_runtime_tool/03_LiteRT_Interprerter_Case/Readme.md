Interpreter API case

- Push the existing aml_delegate_test_bin to an executable directory on the device (for example /data/local/tmp/ or your designated working directory), and make sure it is executable:
  - First, adb push the binary to the target directory
  - Then, on the device, run chmod +x <binary>
- Run command:

```bash
export LD_LIBRARY_PATH=/vendor/lib64
./aml_delegate_test_bin mobilenet_v2_1.0_224_quant.tflite
```

- Example output:

```bash
The minimum amount of memory needed for initializing the context is:
        model alloc from os          0x00000000,( 0.00 MByte)
        model alloc from drv         0x0085ae8c,( 8.36 MByte)
        model addition alloc from os 0x00031700,( 0.19 MByte)
sys mem_available 6004556 KB (5863.82 MB)
Evaluate the smmu_tlb_type is 0
[ADLAU WARN]  [evaluate_addition_mem_usage:5185]
Your model file may contain unnecessary data, which will cause waste of memory.
It is recommended to check your settings when generate the model file.
memory usage:
        model alloc from os          0x00000000,( 0.00 MByte)
        model alloc from drv         0x0085ae8c,( 8.36 MByte)
        model addition alloc from os  0x0004b2c0,( 0.29 MByte)
5.AML_Delegate AllocateTensors
6.AML_Delegate set input
6.AML_Delegate Start to do invoke
Inference status: 0
Output tensor 0
Output tensor name: Identity
Output tensor type: 3
Output tensor size (bytes): 10140
Output tensor dims: 1 2535 4
Output tensor 1
Output tensor name: Identity_1
Output tensor type: 3
Output tensor size (bytes): 202800
Output tensor dims: 1 2535 80
-------------- finished -------------
7.CI test exit (skip interpreter teardown to avoid hang)
```

