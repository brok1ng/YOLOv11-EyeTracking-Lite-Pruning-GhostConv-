  - 1、main分支为剪枝训练代码，主要功能如下：
  
 `step1_train()`：加载预训练模型并开始训练。
  
 `step2_Constraint_train()`：在约束条件下进行训练，例如可能涉及到正则化或其他约束条件。
   
 `step3_pruning()`：使用自定义的`do_pruning`函数对模型进行剪枝，以减少模型的复杂度。
 
 `step4_finetune()`：微调剪枝后的模型。

   - 2、ui分支则为应用模型的小app，主要功能为图片识别、视频跟踪，具体使用方式见ui分支下的readme.
   
演示视频:

https://github.com/user-attachments/assets/b908df7f-4b98-4266-b9dc-67c312bbd781

https://github.com/user-attachments/assets/d7bdbe0f-d70b-41c3-bdf0-940a67fb328c
