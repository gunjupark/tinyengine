## MCU Net 및 Tiny Engine 소스 분석

### 전체 코드 플로우 분석
<br>
* Training 관련
    * QAS , backward , parseTraninable, SGD ...
* Inference 관련
    * Example modules & Flow
        * modules
            * TFLiteConvertor
                * tflite의 convertor가 아닌 자체적인 클래스
                * TFLite 모델의 Operator를 TinyEngine의 basic\_utils.basicOperator를 상속받는
                code\_generator.operators의 Operator(ex. Conv2d, DepthwiseConv2d, Add  등)로 변환
                * 이렇게 변환된 op(TinyEngine's BasicOperator 기반 여러 operator)들은 tflite\_convertor.layer에 리스트 형태로 저장됨
            * TFLiteConvertor's Layer list (예제 상, **tf\_convertor.layer**) 사용되는 곳
                * getPatchParams 및 PatchResizer : TinyEngine의 patch Inference 수행을 위한 파라미터 계산
                * MemoryScheduler 인스턴스 생성시 활용 → 이때 memsche의 멤버변수로 저장됨
                * CodeGenerator클래스 생성시, CodeGenerator의 self.MemSche 멤버변수에 위의 memory scheduler가 저장되며
                **CodeGenerator 인스턴스 내부에서 layer list 정보 필요시, self.MemSche.layer 와 같이 접근하여 활용됨**
            * MemoryScheduler
            \*
    * Gen code flow (아래에서 각 단계별 상세 설명)
        * **Mem buffer → include header → Det process → PatchInference →  Invoke(Inf) → file pointer closing**
    * profiling\_str : 기존 inference\_str 에 디버깅 정보 print 문 추가
* Memory Scheduler
    * General memory scheduler

<br>
<br>
<br>
### Code Generator 소스 분석

* Patch Inference 관련
    * per patch inference 간단 정리
        * 1 \~ N 번째 레이어까지는, 초기 Activation map(feature map) 메모리가 너무 크기 때문에
        layer by layer forward 연산이 불가능 (peak mem 초과)
        * 따라서 코드 상 split layer 이전까지는 per patch inference를 수행하며,
        그 이후로는 per layer 연산을 수행함

* Memory Scheduler 관련
    * 내부 변수 정리
        * buffers : input,output, resiual, im2col, kernel, feature, trainable(w&b) 각각의 텐서 크기를 저장하는 딕셔너리
        * memory\_limit :  기본값으로 10 \*1024 \*1024가 들어감  -
            * Disco 보드는 SRAM 320KB 이 최대지만, TFLM, TTE 의 최대 SRAM 값을 지칭
            * FirstFit에서 메모리 할당 최대 값을 제한하는데 사용됨
        * 
    * 버퍼관리 관련 간단정리
        * allocateMemory 방식 (using scheduler)
            * Inplace Depthwise Conv 별도 처리 (Input, Output Tensor idx 매핑) (+ 학습 시 Transposed conv 고려)
            * outputTable (Parsing된 데이터) 처리 -> 학습시 사용되는 코드
            * Allocate Index 결정 및 할당
            \*
                * Layer를 순회하면서 단일 Op 체크
                    * Op 당 가지고 있는 input tensors, output tensor 를 체크한다.
                    * 위 tensor들의 내부변수인 allocator\_idx를 체크해서 (할당 여부 판단) unallocated\_tensors 리스트에 추가해 둠
                    * unallocated\_tensor들에 대해 addTensor 실행
                    * **baseAllocator:addTensor**
                        * 텐서의 기본적인 정보 (start, end)==(size) , placement(메모리 스케쥴&할당 결과) , **Rectangles 리스트에 append**
                        * **Rectangles = TTE 텐서 실체**
                            * placement = 텐서 배치 (기본값 : -1 , -1일경우 아직 배정되지 않음을 의미)
                            * placement는 MemoryAllocate 함수 내부의
                            allocator.allocate 함수 > firstFit 인스턴스의 fit 함수에서 결정됨
                            * FisrtFit:fit() 의 return 값은 lowest\_slot\_starting
                                * 이 값이 의미하는 바는 Rectangle의 offset. 즉 buffer의 offset을 의미함

<br>
<br>
<br>
<br>
