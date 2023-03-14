# Operator 분석 (High level Kernel name 명세)

Operator 등 다양한 곳에서 사용되는 상수 변수들은 code\_generator/constant.py 에 있음
(ex.  USE\_BIT\_MASK, USE\_TTE\_INT8 등 ...)

Operator의 \_genOpStr 함수 -> generate\_inference\_str() 함수를 통해
invoke를 구성하기도 하지만
**OpGenerator 클래스의 genOpcode를 통해**
Inplace Depthwise Conv Kernel을 직접 생성하기도 한다.
(code\_generator/codetemplate/depthwiseTemplate.py::depthwiseInplace 클래스)

## Base Operator 분석

* class : basic\_utils.py :: basicOperator
* 기본 내부변수로 input\_tensors, outputs\_tensors (list)를 가짐
* self.params를 통해 실제 operation 코드 생성 시 활용
* generate\_str -> CodeGenerator의 \_genOpstr 에서 호출
    * generate\_profiling\_str : 프린트 디버깅용 코드 생성
    * generate\_inference\_str : 추론용 코드만 생성
* tensor 및 메모리 관리용 함수들
    * \_add\_input, \_add\_output 등
    * change\_output\_tensor\_idx : INT8 bias add에서 사용됨
    * getBufferstr : 코드에 반영할 buffer offset string 작성 후 리턴
* 기타 여러가지 변수들에 대한 getter API 제공

<br>
## Operator 종류

TFLite Operation 의 경우
TinyEngine의 CodeGenerator 코드 상에는 지원하는 것 처럼 보이나,
커널의 실체가 구현되어있지 않음 (Interface만 generate 함...)
<br>
### **Conv2d**

* 약 40개 이상의 params 보유 (op\_related, tensor\_related, quantization\_related, fp\_implementation, Q-training, parktial channel update ...)
    * qualtization\_related params (학습에서만 활용되는 QAS 관련 변수들이 있음)
        * effictive\_scale
            * QAS(Quantization Aware Scaling - MCUNet v3 : Training)관련 변수?
            * **일반 추론에도 사용되는 변수임**
            * fp\_requant에서 multiplier 및 shift 계산시 활용
        * zero\_points :  int8 qnn offset
        * scales : int8 qnn scale
        * multiplier , shift : fp requant 관련 변수
    * 학습(v3) 관련 코드 키워드
        * TTEParser : Tiny Training Engine 관련 parser로 추론시에는 활용 X
        * QAS : 마찬가지로 추론에는 상관없음
        * first\_k\_channel : 학습시 partial Channels 사용하도록 설정하는 플래그 변수 (추론 X)
    * (추론에도 쓰임)fp\_requantize : Fully Integer 추론이 가능한지 여부에 따라 False, True로 나뉨
        * ex) VWW = False , Detection = True
* **인스턴스 초기화(Init)**
    * params 가져오기 -> 내부 변수(self.params)에 deep copy
    * 초기화 시에, Input, Output 텐서 추가
        * 이때 사용되는 graph\_idx (input\_idx)는 TFLite Graph상 텐서의 Index를 의미함
        * 텐서 레이아웃 포맷은 N(1) HWC 포맷
* **코드 생성(generate\_inference\_str)**
    * <span style="color:#e11d21">**~~FP32 (TFLite OP Not implemented)~~**</span>
        * TFLite Operation을 사용함
            * 함수명 : TFLite\_Conv\_fp **(tinyengine github 소스코드에는 미포함 된것으로 보임...)**
        * getBufferStr 등의 함수도 INT8(char) 기준이므로, float 시 cast를 수행
            * \_getBuffestrCast이라는 별도 함수를 사용해서 float 배열에 버퍼 할당
    * **<span style="color:#e11d21">~~INT8 TFLite (TFLite OP Not Implemented)~~</span><span style="color:#e11d21"></span>**
        * TFLite\_Conv\_int8\_PerChannel : **마찬가지로 미구현**
    * **INT8 TTE (FP\_output, INT8\_output 모두)**
        * kernel\_h 를 사용하여 1x1 , 3x3 구분
        * **INT\_forward\_op**
            * **1x1 conv**
                * oddch
                    * FP output일경우 : <span style="color:#ce9178">**convolve\_1x1\_s8\_oddch\_fp**</span>
                    * INT8 output일경우 : <span style="color:#ce9178">**convolve\_1x1\_s8\_oddch**</span>
                * channel 수 = 8,16,24,48 일 경우 (aggressive\_unroll , No BitMask)
                    * <span style="color:#ce9178">**convolve\_1x1\_s8\_ch8, 16, 24, 48**</span>
                * 그 외
                    * kbuf , SRAM, skip\_pad ... (코드상 언제 사용되는지 확인 X)
            * **2x3, 3x2, 3x3 conv**
                * (u8은 안쓰이는듯)
                * Patch Inference 시, 커널이름 앞에 "patchpadding\_" 이 추가됨
                (ex. <span style="color:#ce9178">**patchpadding\_convolve\_s8\_kernel3\_inputch3\_stride2** <span style="color:#000000">)</span></span>
                * <span style="color:#ce9178"><span style="color:#000000">General Case : </span>**convolve\_s8\_kernel3**</span><span style="color:#000000">(\_strideX\_padX)
                (ex. </span><span style="color:#ce9178">**convolve\_s8\_kernel3\_inputch3\_stride2\_pad1** </span><span style="color:#000000">)</span>
        * **FP\_requantize\_op (fp\_requantize = True일 경우)**
            * TinyEngine/src/kernels/fp\_requantize\_op 에 있는 함수를 사용
            * 위의 INT forward Op 커널들에 대해 "\_freq" 가 추가된 커널임
            * BitMask 필요 유무(need\_Bmask 변수)에 따라 "\_mask" 추가된 커널사용 **(학습 전용??)**
        * 아래와 같이 kbuf가 추가되는 케이스가 있음
        (생성된 코드 확인 결과 : 첫번째 Conv 레이어만 적용되는것으로 확인됨)

``` python
if kernel_h == 3 and params["stride_h"] == 2 and params["padding"] == 1:
   string += ",kbuf"
```

* Patch Inference시, pad\_t , pad\_b, pad\_l , pad\_r 변수를 매개변수에 추가
* hard\_switsh(swish?) Activation Function 사용 시, TFLite 함수를 가져다가 씀
( <span>tflite::reference\_ops::HardSwish<int8\_t>)</int8\_t></span>
* 
*

<br>
### Depthwise Conv 2D

* **FP32, INT8 TFLite operation : 미구현**
*

<br>
### Add
<br>
<br>
### Average Pooling
<br>
### SE Module 처리 방식
<br>
<br>
<br>
(Detection Part)

* 모든 레이어 fp\_requantize  =>  커널명에 \_freq 붙음
* Detection Post Process 함수 추가
* no Patch Inference
* Upsample, Max Pooling (for FPN) 추가

### Upsample
<br>
### Max Pool
<br>
### Detection post process
<br>
<br>
<br>
## Examples - Generated Code 분석

### 1\. VWW\(visual wake word\)\_patchbased
<br>
<br>
<br>
### 2\. Detection\_fpn
<br>
<br>
