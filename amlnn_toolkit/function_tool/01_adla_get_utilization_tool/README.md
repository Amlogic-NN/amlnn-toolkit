# 1.使用方法

## 1.1 板子连接adb, 插上网线
## 1.2 打开Powershell窗口执行 adb root & adb shell & ifconfig(获取ip)
## 1.3 PC服务器端执行
1) 使用方式一(Linux):
adb connect 10.18.9.126 (ip)
python3 NPU_utilization.py

2) 使用方式二(winndows):
python NPU_utilization.py

# 2.显示信息如下

![alt text](npu_utilization_detail.png)

