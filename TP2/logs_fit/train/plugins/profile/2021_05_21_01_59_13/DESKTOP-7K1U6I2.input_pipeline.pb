	?f??67???f??67??!?f??67??	???#B?!@???#B?!@!???#B?!@"{
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails:?f??67???? >????A??v?ӂ??Y{???ί?rEagerKernelExecute 0*	~j?t??T@2U
Iterator::Model::ParallelMapV2???Đ???!U!??@@)???Đ???1U!??@@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat
?2?&??!??C:?F;@)x????Փ?18??_7@:Preprocessing2F
Iterator::Model??a?ã?!k?!?HG@)???SVӅ?1W.=(?)@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate?+J	????!???	yk/@)??2?68??1?,Go?I$@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip9`W?????!?X??4?J@)|DL?$zy?1?6?:?@:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice?.??r?!?+?4/C@)?.??r?1?+?4/C@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor??c${?j?!?È?`>@)??c${?j?1?È?`>@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap?J??q???!?5??p?2@)?{?i??c?1٣Rϣ?@:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is MODERATELY input-bound because 8.9% of the total step time sampled is waiting for input. Therefore, you would need to reduce both the input time and other time.no*high2t20.1 % of the total step time sampled is spent on 'All Others' time. This could be due to Python execution overhead.9???#B?!@I?m??W?V@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	?? >?????? >????!?? >????      ??!       "      ??!       *      ??!       2	??v?ӂ????v?ӂ??!??v?ӂ??:      ??!       B      ??!       J	{???ί?{???ί?!{???ί?R      ??!       Z	{???ί?{???ί?!{???ί?b      ??!       JCPU_ONLYY???#B?!@b q?m??W?V@