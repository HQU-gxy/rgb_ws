## 控制板

ESP32具有以下特性

处理器：
- CPU: Xtensa 双核 (或单核) 32位 LX6 微处理器, 时钟速度 160/240 MHz, 算力高达 600 DMIPS

存储：
- 448 KB ROM (64KB+384KB)， 520 KB SRAM

无线连接:
- Wi-Fi: 802.11 b/g/n
- 蓝牙: v4.2 BR/EDR 和 BLE (和Wi-Fi共享射频模块)

外设接口:

- 34个可编程 GPIOs
- 12位SAR ADC，高达18个通道
- 2个8位 DAC
- 10个触摸传感器(电容式感应 GPIOs)
- 4 × SPI
- 2 × I²S 接口
- 2 × I²C 接口
- 3 × UART
- SD/SDIO/CE-ATA/MMC/eMMC 主控制器
- SDIO/SPI 从控制器
- 具有专用DMA和计划支持 IEEE精准时间协议[4] 的以太网接口。
- 控制器局域网 2.0
- 红外控制器 (TX/RX, 多达8通道)
- 脉冲计数器 (支持 全正交 解码)
- 电机 PWM
- LED PWM (多达16通道)
- 超低功耗模拟前置放大器
- 安全特性:
- 支持全部 IEEE 802.11 标准安全功能，包括 WPA, WPA2, WPA3（取决于版本）[5] 以及 无线局域网鉴别与保密基础结构 (WAPI)
- 安全启动
- ROM加密
- 1024位 OTP
- 硬件加密加速: AES, SHA-2, RSA, ECC, 随机数生成 (RNG)
- 电源管理:
- 内部 低压差稳压器
- RTC独立电源域
- 5 μA 深度睡眠电流
- 从GPIO中断, 定时器, ADC , 电容式触摸传感器中断唤醒

## 中继器

ESP32-C3

- NodeMCU 开发板，核心为 ESP32-C3-32S
- 单核32位 RISC-V 微处理器, 高达 160 MHz
- 400 KiB SRAM, 384 KiB ROM, 8 KiB RTC SRAM
- WiFi 2.4 GHz (IEEE 802.11b/g/n)
- Bluetooth 5.0 (低功耗蓝牙)
- 22 个 可编程GPIO
- 2 个 12位ADC
- 和ESP8266 引脚兼容

## 通信

LLCC68 LoRa智能家居（LLCC68）是一款用于中等距离室内及室内到室外无线应用的低频LoRa®射频收发器。支持SPI接口。与SX1262引脚兼容。SX1261、SX1262、SX1268和LLCC68都旨在实现长电池寿命，其活动接收电流消耗仅为4.2mA。SX1261可以传输高达+15dBm的功率，而SX1262、SX1268和LLCC68可以传输高达+22dBm的功率，并且集成了高效的功率放大器。

这些设备支持用于LPWAN用例的LoRa调制和用于传统用例的(G)FSK调制。这些设备高度可配置，以满足消费者使用的不同应用需求。设备提供与Semtech收发器兼容的LoRa调制，这些收发器由LoRa Alliance®发布的LoRaWAN®规范使用。该无线电适用于旨在符合无线电规定的系统，包括但不限于ETSI EN 300 220、FCC CFR 47第15部分、中国法规要求和日本ARIB T-108。从150MHz到960MHz的连续频率覆盖允许支持世界各地所有主要的次GHz ISM频段。

特点
- LoRa和FSK调制解调器
- 最高151dB的链路预算（LLCC68）
- +22dBm或+15dBm高效PA
- 低接收电流4.6 mA
- 集成的DC-DC转换器和LDO
- 可编程比特率，从1.76kbps到62.5kbps LoRa和300kbps FSK
- 高灵敏度：低至-129dBm
- 在1MHz偏移处88dB的阻塞抑制
- 在LoRa模式下19dB的同频道拒绝
- 支持FSK、GFSK、MSK、GMSK和LoRa调制
- 内置位同步器用于时钟恢复
- 具有超快AFC的自动信道活动检测（CAD）
- LoRaWAN的数据速率包括在125kHz带宽下扩频因子SF7至SF9、在250kHz带宽下扩频因子SF7至SF10，以及在500kHz带宽下扩频因子SF7至SF11
