Operator 분석 (High level Kernel name 명세)
Operator 등 다양한 곳에서 사용되는 상수 변수들은 code_generator/constant.py 에 있음
(ex.  USE_BIT_MASK, USE_TTE_INT8 등 ...)


Operator의 _genOpStr 함수 -> generate_inference_str() 함수를 통해
invoke를 구성하기도 하지만
OpGenerator 클래스의 genOpcode를 통해
Inplace Depthwise Conv Kernel을 직접 생성하기도 한다.
(code_generator/codetemplate/depthwiseTemplate.py::depthwiseInplace 클래스)
Base Operator 분석
class : basic_utils.py :: basicOperator
기본 내부변수로 input_tensors, outputs_tensors (list)를 가짐
self.params를 통해 실제 operation 코드 생성 시 활용
generate_str -> CodeGenerator의 _genOpstr 에서 호출
generate_profiling_str : 프린트 디버깅용 코드 생성
generate_inference_str : 추론용 코드만 생성
tensor 및 메모리 관리용 함수들
_add_input, _add_output 등
change_output_tensor_idx : INT8 bias add에서 사용됨
getBufferstr : 코드에 반영할 buffer offset string 작성 후 리턴
기타 여러가지 변수들에 대한 getter API 제공


Operator 종류
TFLite Operation 의 경우
TinyEngine의 CodeGenerator 코드 상에는 지원하는 것 처럼 보이나,
커널의 실체가 구현되어있지 않음 (Interface만 generate 함...)


Conv2d
약 40개 이상의 params 보유 (op_related, tensor_related, quantization_related, fp_implementation, Q-training, parktial channel update ...)
qualtization_related params (학습에서만 활용되는 QAS 관련 변수들이 있음)
effictive_scale
QAS(Quantization Aware Scaling - MCUNet v3 : Training)관련 변수?
일반 추론에도 사용되는 변수임
fp_requant에서 multiplier 및 shift 계산시 활용
zero_points :  int8 qnn offset
scales : int8 qnn scale
multiplier , shift : fp requant 관련 변수
학습(v3) 관련 코드 키워드
TTEParser : Tiny Training Engine 관련 parser로 추론시에는 활용 X
QAS : 마찬가지로 추론에는 상관없음
first_k_channel : 학습시 partial Channels 사용하도록 설정하는 플래그 변수 (추론 X)
(추론에도 쓰임)fp_requantize : Fully Integer 추론이 가능한지 여부에 따라 False, True로 나뉨
ex) VWW = False , Detection = True
인스턴스 초기화(Init)
params 가져오기 -> 내부 변수(self.params)에 deep copy
초기화 시에, Input, Output 텐서 추가
이때 사용되는 graph_idx (input_idx)는 TFLite Graph상 텐서의 Index를 의미함
텐서 레이아웃 포맷은 N(1) HWC 포맷
코드 생성(generate_inference_str)
FP32 (TFLite OP Not implemented)
TFLite Operation을 사용함
함수명 : TFLite_Conv_fp (tinyengine github 소스코드에는 미포함 된것으로 보임...)
getBufferStr 등의 함수도 INT8(char) 기준이므로, float 시 cast를 수행
_getBuffestrCast이라는 별도 함수를 사용해서 float 배열에 버퍼 할당
INT8 TFLite (TFLite OP Not Implemented)
TFLite_Conv_int8_PerChannel : 마찬가지로 미구현
INT8 TTE (FP_output, INT8_output 모두)
kernel_h 를 사용하여 1x1 , 3x3 구분
INT_forward_op
1x1 conv
oddch
FP output일경우 : convolve_1x1_s8_oddch_fp
INT8 output일경우 : convolve_1x1_s8_oddch
channel 수 = 8,16,24,48 일 경우 (aggressive_unroll , No BitMask)
convolve_1x1_s8_ch8, 16, 24, 48
그 외
kbuf , SRAM, skip_pad ... (코드상 언제 사용되는지 확인 X)
2x3, 3x2, 3x3 conv
(u8은 안쓰이는듯)
Patch Inference 시, 커널이름 앞에 "patchpadding_" 이 추가됨
(ex. patchpadding_convolve_s8_kernel3_inputch3_stride2 )
General Case : convolve_s8_kernel3(_strideX_padX)
(ex. convolve_s8_kernel3_inputch3_stride2_pad1 )
FP_requantize_op (fp_requantize = True일 경우)
TinyEngine/src/kernels/fp_requantize_op 에 있는 함수를 사용
위의 INT forward Op 커널들에 대해 "_freq" 가 추가된 커널임
BitMask 필요 유무(need_Bmask 변수)에 따라 "_mask" 추가된 커널사용 (학습 전용??)
아래와 같이 kbuf가 추가되는 케이스가 있음
(생성된 코드 확인 결과 : 첫번째 Conv 레이어만 적용되는것으로 확인됨)
if kernel_h == 3 and params["stride_h"] == 2 and params["padding"] == 1:
   string += ",kbuf"
Patch Inference시, pad_t , pad_b, pad_l , pad_r 변수를 매개변수에 추가
hard_switsh(swish?) Activation Function 사용 시, TFLite 함수를 가져다가 씀
( tflite::reference_ops::HardSwish<int8_t>)</int8_t>






Depthwise Conv 2D
FP32, INT8 TFLite operation : 미구현




Add




Average Pooling


SE Module 처리 방식






(Detection Part)
모든 레이어 fp_requantize  =>  커널명에 _freq 붙음
Detection Post Process 함수 추가
no Patch Inference
Upsample, Max Pooling (for FPN) 추가
Upsample


Max Pool


Detection post process






Examples - Generated Code 분석
1. VWW(visual wake word)_patchbased






2. Detection_fpn



