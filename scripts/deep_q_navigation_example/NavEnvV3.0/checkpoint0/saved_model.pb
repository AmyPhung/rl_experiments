Ģö
Ķ£
8
Const
output"dtype"
valuetensor"
dtypetype

NoOp
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
@
ReadVariableOp
resource
value"dtype"
dtypetype
¾
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring 

VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 "serve*2.3.12v2.3.0-54-gfcc4b966f18£

basic_dqn/dense_2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
: *)
shared_namebasic_dqn/dense_2/kernel

,basic_dqn/dense_2/kernel/Read/ReadVariableOpReadVariableOpbasic_dqn/dense_2/kernel*
_output_shapes

: *
dtype0

basic_dqn/dense_2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *'
shared_namebasic_dqn/dense_2/bias
}
*basic_dqn/dense_2/bias/Read/ReadVariableOpReadVariableOpbasic_dqn/dense_2/bias*
_output_shapes
: *
dtype0

basic_dqn/dense_3/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:  *)
shared_namebasic_dqn/dense_3/kernel

,basic_dqn/dense_3/kernel/Read/ReadVariableOpReadVariableOpbasic_dqn/dense_3/kernel*
_output_shapes

:  *
dtype0

basic_dqn/dense_3/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *'
shared_namebasic_dqn/dense_3/bias
}
*basic_dqn/dense_3/bias/Read/ReadVariableOpReadVariableOpbasic_dqn/dense_3/bias*
_output_shapes
: *
dtype0

basic_dqn/q_values/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
: **
shared_namebasic_dqn/q_values/kernel

-basic_dqn/q_values/kernel/Read/ReadVariableOpReadVariableOpbasic_dqn/q_values/kernel*
_output_shapes

: *
dtype0

basic_dqn/q_values/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*(
shared_namebasic_dqn/q_values/bias

+basic_dqn/q_values/bias/Read/ReadVariableOpReadVariableOpbasic_dqn/q_values/bias*
_output_shapes
:*
dtype0
f
	Adam/iterVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *
shared_name	Adam/iter
_
Adam/iter/Read/ReadVariableOpReadVariableOp	Adam/iter*
_output_shapes
: *
dtype0	
j
Adam/beta_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_1
c
Adam/beta_1/Read/ReadVariableOpReadVariableOpAdam/beta_1*
_output_shapes
: *
dtype0
j
Adam/beta_2VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_2
c
Adam/beta_2/Read/ReadVariableOpReadVariableOpAdam/beta_2*
_output_shapes
: *
dtype0
h

Adam/decayVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name
Adam/decay
a
Adam/decay/Read/ReadVariableOpReadVariableOp
Adam/decay*
_output_shapes
: *
dtype0
x
Adam/learning_rateVarHandleOp*
_output_shapes
: *
dtype0*
shape: *#
shared_nameAdam/learning_rate
q
&Adam/learning_rate/Read/ReadVariableOpReadVariableOpAdam/learning_rate*
_output_shapes
: *
dtype0
^
totalVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nametotal
W
total/Read/ReadVariableOpReadVariableOptotal*
_output_shapes
: *
dtype0
^
countVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namecount
W
count/Read/ReadVariableOpReadVariableOpcount*
_output_shapes
: *
dtype0

Adam/basic_dqn/dense_2/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
: *0
shared_name!Adam/basic_dqn/dense_2/kernel/m

3Adam/basic_dqn/dense_2/kernel/m/Read/ReadVariableOpReadVariableOpAdam/basic_dqn/dense_2/kernel/m*
_output_shapes

: *
dtype0

Adam/basic_dqn/dense_2/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *.
shared_nameAdam/basic_dqn/dense_2/bias/m

1Adam/basic_dqn/dense_2/bias/m/Read/ReadVariableOpReadVariableOpAdam/basic_dqn/dense_2/bias/m*
_output_shapes
: *
dtype0

Adam/basic_dqn/dense_3/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:  *0
shared_name!Adam/basic_dqn/dense_3/kernel/m

3Adam/basic_dqn/dense_3/kernel/m/Read/ReadVariableOpReadVariableOpAdam/basic_dqn/dense_3/kernel/m*
_output_shapes

:  *
dtype0

Adam/basic_dqn/dense_3/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *.
shared_nameAdam/basic_dqn/dense_3/bias/m

1Adam/basic_dqn/dense_3/bias/m/Read/ReadVariableOpReadVariableOpAdam/basic_dqn/dense_3/bias/m*
_output_shapes
: *
dtype0

 Adam/basic_dqn/q_values/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
: *1
shared_name" Adam/basic_dqn/q_values/kernel/m

4Adam/basic_dqn/q_values/kernel/m/Read/ReadVariableOpReadVariableOp Adam/basic_dqn/q_values/kernel/m*
_output_shapes

: *
dtype0

Adam/basic_dqn/q_values/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*/
shared_name Adam/basic_dqn/q_values/bias/m

2Adam/basic_dqn/q_values/bias/m/Read/ReadVariableOpReadVariableOpAdam/basic_dqn/q_values/bias/m*
_output_shapes
:*
dtype0

Adam/basic_dqn/dense_2/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
: *0
shared_name!Adam/basic_dqn/dense_2/kernel/v

3Adam/basic_dqn/dense_2/kernel/v/Read/ReadVariableOpReadVariableOpAdam/basic_dqn/dense_2/kernel/v*
_output_shapes

: *
dtype0

Adam/basic_dqn/dense_2/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *.
shared_nameAdam/basic_dqn/dense_2/bias/v

1Adam/basic_dqn/dense_2/bias/v/Read/ReadVariableOpReadVariableOpAdam/basic_dqn/dense_2/bias/v*
_output_shapes
: *
dtype0

Adam/basic_dqn/dense_3/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:  *0
shared_name!Adam/basic_dqn/dense_3/kernel/v

3Adam/basic_dqn/dense_3/kernel/v/Read/ReadVariableOpReadVariableOpAdam/basic_dqn/dense_3/kernel/v*
_output_shapes

:  *
dtype0

Adam/basic_dqn/dense_3/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *.
shared_nameAdam/basic_dqn/dense_3/bias/v

1Adam/basic_dqn/dense_3/bias/v/Read/ReadVariableOpReadVariableOpAdam/basic_dqn/dense_3/bias/v*
_output_shapes
: *
dtype0

 Adam/basic_dqn/q_values/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
: *1
shared_name" Adam/basic_dqn/q_values/kernel/v

4Adam/basic_dqn/q_values/kernel/v/Read/ReadVariableOpReadVariableOp Adam/basic_dqn/q_values/kernel/v*
_output_shapes

: *
dtype0

Adam/basic_dqn/q_values/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*/
shared_name Adam/basic_dqn/q_values/bias/v

2Adam/basic_dqn/q_values/bias/v/Read/ReadVariableOpReadVariableOpAdam/basic_dqn/q_values/bias/v*
_output_shapes
:*
dtype0

NoOpNoOp
·!
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*ņ 
valueč Bå  BŽ 

fc1
fc2

logits
	optimizer
	variables
regularization_losses
trainable_variables
	keras_api
	
signatures
h


kernel
bias
	variables
regularization_losses
trainable_variables
	keras_api
h

kernel
bias
	variables
regularization_losses
trainable_variables
	keras_api
h

kernel
bias
	variables
regularization_losses
trainable_variables
	keras_api
¬
iter

beta_1

beta_2
	decay
 learning_rate
m:m;m<m=m>m?
v@vAvBvCvDvE
*

0
1
2
3
4
5
 
*

0
1
2
3
4
5
­

!layers
"metrics
	variables
#layer_metrics
$layer_regularization_losses
%non_trainable_variables
regularization_losses
trainable_variables
 
SQ
VARIABLE_VALUEbasic_dqn/dense_2/kernel%fc1/kernel/.ATTRIBUTES/VARIABLE_VALUE
OM
VARIABLE_VALUEbasic_dqn/dense_2/bias#fc1/bias/.ATTRIBUTES/VARIABLE_VALUE


0
1
 


0
1
­

&layers
'metrics
(layer_metrics
	variables
)layer_regularization_losses
*non_trainable_variables
regularization_losses
trainable_variables
SQ
VARIABLE_VALUEbasic_dqn/dense_3/kernel%fc2/kernel/.ATTRIBUTES/VARIABLE_VALUE
OM
VARIABLE_VALUEbasic_dqn/dense_3/bias#fc2/bias/.ATTRIBUTES/VARIABLE_VALUE

0
1
 

0
1
­

+layers
,metrics
-layer_metrics
	variables
.layer_regularization_losses
/non_trainable_variables
regularization_losses
trainable_variables
WU
VARIABLE_VALUEbasic_dqn/q_values/kernel(logits/kernel/.ATTRIBUTES/VARIABLE_VALUE
SQ
VARIABLE_VALUEbasic_dqn/q_values/bias&logits/bias/.ATTRIBUTES/VARIABLE_VALUE

0
1
 

0
1
­

0layers
1metrics
2layer_metrics
	variables
3layer_regularization_losses
4non_trainable_variables
regularization_losses
trainable_variables
HF
VARIABLE_VALUE	Adam/iter)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEAdam/beta_1+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEAdam/beta_2+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUE
JH
VARIABLE_VALUE
Adam/decay*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUE
ZX
VARIABLE_VALUEAdam/learning_rate2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUE

0
1
2

50
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
4
	6total
	7count
8	variables
9	keras_api
OM
VARIABLE_VALUEtotal4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE
OM
VARIABLE_VALUEcount4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE

60
71

8	variables
vt
VARIABLE_VALUEAdam/basic_dqn/dense_2/kernel/mAfc1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
rp
VARIABLE_VALUEAdam/basic_dqn/dense_2/bias/m?fc1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
vt
VARIABLE_VALUEAdam/basic_dqn/dense_3/kernel/mAfc2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
rp
VARIABLE_VALUEAdam/basic_dqn/dense_3/bias/m?fc2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUE Adam/basic_dqn/q_values/kernel/mDlogits/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
vt
VARIABLE_VALUEAdam/basic_dqn/q_values/bias/mBlogits/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
vt
VARIABLE_VALUEAdam/basic_dqn/dense_2/kernel/vAfc1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
rp
VARIABLE_VALUEAdam/basic_dqn/dense_2/bias/v?fc1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
vt
VARIABLE_VALUEAdam/basic_dqn/dense_3/kernel/vAfc2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
rp
VARIABLE_VALUEAdam/basic_dqn/dense_3/bias/v?fc2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUE Adam/basic_dqn/q_values/kernel/vDlogits/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
vt
VARIABLE_VALUEAdam/basic_dqn/q_values/bias/vBlogits/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
z
serving_default_input_1Placeholder*'
_output_shapes
:’’’’’’’’’*
dtype0*
shape:’’’’’’’’’
Ų
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_1basic_dqn/dense_2/kernelbasic_dqn/dense_2/biasbasic_dqn/dense_3/kernelbasic_dqn/dense_3/biasbasic_dqn/q_values/kernelbasic_dqn/q_values/bias*
Tin
	2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:’’’’’’’’’*(
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8 *,
f'R%
#__inference_signature_wrapper_88596
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 

StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename,basic_dqn/dense_2/kernel/Read/ReadVariableOp*basic_dqn/dense_2/bias/Read/ReadVariableOp,basic_dqn/dense_3/kernel/Read/ReadVariableOp*basic_dqn/dense_3/bias/Read/ReadVariableOp-basic_dqn/q_values/kernel/Read/ReadVariableOp+basic_dqn/q_values/bias/Read/ReadVariableOpAdam/iter/Read/ReadVariableOpAdam/beta_1/Read/ReadVariableOpAdam/beta_2/Read/ReadVariableOpAdam/decay/Read/ReadVariableOp&Adam/learning_rate/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOp3Adam/basic_dqn/dense_2/kernel/m/Read/ReadVariableOp1Adam/basic_dqn/dense_2/bias/m/Read/ReadVariableOp3Adam/basic_dqn/dense_3/kernel/m/Read/ReadVariableOp1Adam/basic_dqn/dense_3/bias/m/Read/ReadVariableOp4Adam/basic_dqn/q_values/kernel/m/Read/ReadVariableOp2Adam/basic_dqn/q_values/bias/m/Read/ReadVariableOp3Adam/basic_dqn/dense_2/kernel/v/Read/ReadVariableOp1Adam/basic_dqn/dense_2/bias/v/Read/ReadVariableOp3Adam/basic_dqn/dense_3/kernel/v/Read/ReadVariableOp1Adam/basic_dqn/dense_3/bias/v/Read/ReadVariableOp4Adam/basic_dqn/q_values/kernel/v/Read/ReadVariableOp2Adam/basic_dqn/q_values/bias/v/Read/ReadVariableOpConst*&
Tin
2	*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *'
f"R 
__inference__traced_save_88753
 
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamebasic_dqn/dense_2/kernelbasic_dqn/dense_2/biasbasic_dqn/dense_3/kernelbasic_dqn/dense_3/biasbasic_dqn/q_values/kernelbasic_dqn/q_values/bias	Adam/iterAdam/beta_1Adam/beta_2
Adam/decayAdam/learning_ratetotalcountAdam/basic_dqn/dense_2/kernel/mAdam/basic_dqn/dense_2/bias/mAdam/basic_dqn/dense_3/kernel/mAdam/basic_dqn/dense_3/bias/m Adam/basic_dqn/q_values/kernel/mAdam/basic_dqn/q_values/bias/mAdam/basic_dqn/dense_2/kernel/vAdam/basic_dqn/dense_2/bias/vAdam/basic_dqn/dense_3/kernel/vAdam/basic_dqn/dense_3/bias/v Adam/basic_dqn/q_values/kernel/vAdam/basic_dqn/q_values/bias/v*%
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 **
f%R#
!__inference__traced_restore_88838ß¦
ć
»
)__inference_basic_dqn_layer_call_fn_88569
input_1
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
identity¢StatefulPartitionedCall¬
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:’’’’’’’’’*(
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_basic_dqn_layer_call_and_return_conditional_losses_885512
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:’’’’’’’’’2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:’’’’’’’’’::::::22
StatefulPartitionedCallStatefulPartitionedCall:P L
'
_output_shapes
:’’’’’’’’’
!
_user_specified_name	input_1
Ø
æ
D__inference_basic_dqn_layer_call_and_return_conditional_losses_88551
input_1
dense_2_88492
dense_2_88494
dense_3_88519
dense_3_88521
q_values_88545
q_values_88547
identity¢dense_2/StatefulPartitionedCall¢dense_3/StatefulPartitionedCall¢ q_values/StatefulPartitionedCall
dense_2/StatefulPartitionedCallStatefulPartitionedCallinput_1dense_2_88492dense_2_88494*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:’’’’’’’’’ *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *K
fFRD
B__inference_dense_2_layer_call_and_return_conditional_losses_884812!
dense_2/StatefulPartitionedCall±
dense_3/StatefulPartitionedCallStatefulPartitionedCall(dense_2/StatefulPartitionedCall:output:0dense_3_88519dense_3_88521*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:’’’’’’’’’ *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *K
fFRD
B__inference_dense_3_layer_call_and_return_conditional_losses_885082!
dense_3/StatefulPartitionedCall¶
 q_values/StatefulPartitionedCallStatefulPartitionedCall(dense_3/StatefulPartitionedCall:output:0q_values_88545q_values_88547*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:’’’’’’’’’*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_q_values_layer_call_and_return_conditional_losses_885342"
 q_values/StatefulPartitionedCallä
IdentityIdentity)q_values/StatefulPartitionedCall:output:0 ^dense_2/StatefulPartitionedCall ^dense_3/StatefulPartitionedCall!^q_values/StatefulPartitionedCall*
T0*'
_output_shapes
:’’’’’’’’’2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:’’’’’’’’’::::::2B
dense_2/StatefulPartitionedCalldense_2/StatefulPartitionedCall2B
dense_3/StatefulPartitionedCalldense_3/StatefulPartitionedCall2D
 q_values/StatefulPartitionedCall q_values/StatefulPartitionedCall:P L
'
_output_shapes
:’’’’’’’’’
!
_user_specified_name	input_1
Ż
}
(__inference_q_values_layer_call_fn_88655

inputs
unknown
	unknown_0
identity¢StatefulPartitionedCallö
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:’’’’’’’’’*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_q_values_layer_call_and_return_conditional_losses_885342
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:’’’’’’’’’2

Identity"
identityIdentity:output:0*.
_input_shapes
:’’’’’’’’’ ::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:’’’’’’’’’ 
 
_user_specified_nameinputs
Ū
|
'__inference_dense_3_layer_call_fn_88636

inputs
unknown
	unknown_0
identity¢StatefulPartitionedCallõ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:’’’’’’’’’ *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *K
fFRD
B__inference_dense_3_layer_call_and_return_conditional_losses_885082
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:’’’’’’’’’ 2

Identity"
identityIdentity:output:0*.
_input_shapes
:’’’’’’’’’ ::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:’’’’’’’’’ 
 
_user_specified_nameinputs
Ū
|
'__inference_dense_2_layer_call_fn_88616

inputs
unknown
	unknown_0
identity¢StatefulPartitionedCallõ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:’’’’’’’’’ *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *K
fFRD
B__inference_dense_2_layer_call_and_return_conditional_losses_884812
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:’’’’’’’’’ 2

Identity"
identityIdentity:output:0*.
_input_shapes
:’’’’’’’’’::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:’’’’’’’’’
 
_user_specified_nameinputs
Õk

!__inference__traced_restore_88838
file_prefix-
)assignvariableop_basic_dqn_dense_2_kernel-
)assignvariableop_1_basic_dqn_dense_2_bias/
+assignvariableop_2_basic_dqn_dense_3_kernel-
)assignvariableop_3_basic_dqn_dense_3_bias0
,assignvariableop_4_basic_dqn_q_values_kernel.
*assignvariableop_5_basic_dqn_q_values_bias 
assignvariableop_6_adam_iter"
assignvariableop_7_adam_beta_1"
assignvariableop_8_adam_beta_2!
assignvariableop_9_adam_decay*
&assignvariableop_10_adam_learning_rate
assignvariableop_11_total
assignvariableop_12_count7
3assignvariableop_13_adam_basic_dqn_dense_2_kernel_m5
1assignvariableop_14_adam_basic_dqn_dense_2_bias_m7
3assignvariableop_15_adam_basic_dqn_dense_3_kernel_m5
1assignvariableop_16_adam_basic_dqn_dense_3_bias_m8
4assignvariableop_17_adam_basic_dqn_q_values_kernel_m6
2assignvariableop_18_adam_basic_dqn_q_values_bias_m7
3assignvariableop_19_adam_basic_dqn_dense_2_kernel_v5
1assignvariableop_20_adam_basic_dqn_dense_2_bias_v7
3assignvariableop_21_adam_basic_dqn_dense_3_kernel_v5
1assignvariableop_22_adam_basic_dqn_dense_3_bias_v8
4assignvariableop_23_adam_basic_dqn_q_values_kernel_v6
2assignvariableop_24_adam_basic_dqn_q_values_bias_v
identity_26¢AssignVariableOp¢AssignVariableOp_1¢AssignVariableOp_10¢AssignVariableOp_11¢AssignVariableOp_12¢AssignVariableOp_13¢AssignVariableOp_14¢AssignVariableOp_15¢AssignVariableOp_16¢AssignVariableOp_17¢AssignVariableOp_18¢AssignVariableOp_19¢AssignVariableOp_2¢AssignVariableOp_20¢AssignVariableOp_21¢AssignVariableOp_22¢AssignVariableOp_23¢AssignVariableOp_24¢AssignVariableOp_3¢AssignVariableOp_4¢AssignVariableOp_5¢AssignVariableOp_6¢AssignVariableOp_7¢AssignVariableOp_8¢AssignVariableOp_9
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*
valueBB%fc1/kernel/.ATTRIBUTES/VARIABLE_VALUEB#fc1/bias/.ATTRIBUTES/VARIABLE_VALUEB%fc2/kernel/.ATTRIBUTES/VARIABLE_VALUEB#fc2/bias/.ATTRIBUTES/VARIABLE_VALUEB(logits/kernel/.ATTRIBUTES/VARIABLE_VALUEB&logits/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEBAfc1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEB?fc1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBAfc2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEB?fc2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBDlogits/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBlogits/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBAfc1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB?fc1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBAfc2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB?fc2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBDlogits/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBlogits/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2/tensor_namesĀ
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*G
value>B<B B B B B B B B B B B B B B B B B B B B B B B B B B 2
RestoreV2/shape_and_slices­
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*|
_output_shapesj
h::::::::::::::::::::::::::*(
dtypes
2	2
	RestoreV2g
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:2

IdentityØ
AssignVariableOpAssignVariableOp)assignvariableop_basic_dqn_dense_2_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1®
AssignVariableOp_1AssignVariableOp)assignvariableop_1_basic_dqn_dense_2_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1k

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:2

Identity_2°
AssignVariableOp_2AssignVariableOp+assignvariableop_2_basic_dqn_dense_3_kernelIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_2k

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:2

Identity_3®
AssignVariableOp_3AssignVariableOp)assignvariableop_3_basic_dqn_dense_3_biasIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_3k

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:2

Identity_4±
AssignVariableOp_4AssignVariableOp,assignvariableop_4_basic_dqn_q_values_kernelIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_4k

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:2

Identity_5Æ
AssignVariableOp_5AssignVariableOp*assignvariableop_5_basic_dqn_q_values_biasIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_5k

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0	*
_output_shapes
:2

Identity_6”
AssignVariableOp_6AssignVariableOpassignvariableop_6_adam_iterIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	2
AssignVariableOp_6k

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:2

Identity_7£
AssignVariableOp_7AssignVariableOpassignvariableop_7_adam_beta_1Identity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_7k

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:2

Identity_8£
AssignVariableOp_8AssignVariableOpassignvariableop_8_adam_beta_2Identity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_8k

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:2

Identity_9¢
AssignVariableOp_9AssignVariableOpassignvariableop_9_adam_decayIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_9n
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:2
Identity_10®
AssignVariableOp_10AssignVariableOp&assignvariableop_10_adam_learning_rateIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_10n
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:2
Identity_11”
AssignVariableOp_11AssignVariableOpassignvariableop_11_totalIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_11n
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:2
Identity_12”
AssignVariableOp_12AssignVariableOpassignvariableop_12_countIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_12n
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:2
Identity_13»
AssignVariableOp_13AssignVariableOp3assignvariableop_13_adam_basic_dqn_dense_2_kernel_mIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_13n
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:2
Identity_14¹
AssignVariableOp_14AssignVariableOp1assignvariableop_14_adam_basic_dqn_dense_2_bias_mIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_14n
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:2
Identity_15»
AssignVariableOp_15AssignVariableOp3assignvariableop_15_adam_basic_dqn_dense_3_kernel_mIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_15n
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:2
Identity_16¹
AssignVariableOp_16AssignVariableOp1assignvariableop_16_adam_basic_dqn_dense_3_bias_mIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_16n
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:2
Identity_17¼
AssignVariableOp_17AssignVariableOp4assignvariableop_17_adam_basic_dqn_q_values_kernel_mIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_17n
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:2
Identity_18ŗ
AssignVariableOp_18AssignVariableOp2assignvariableop_18_adam_basic_dqn_q_values_bias_mIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_18n
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:2
Identity_19»
AssignVariableOp_19AssignVariableOp3assignvariableop_19_adam_basic_dqn_dense_2_kernel_vIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_19n
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:2
Identity_20¹
AssignVariableOp_20AssignVariableOp1assignvariableop_20_adam_basic_dqn_dense_2_bias_vIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_20n
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:2
Identity_21»
AssignVariableOp_21AssignVariableOp3assignvariableop_21_adam_basic_dqn_dense_3_kernel_vIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_21n
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:2
Identity_22¹
AssignVariableOp_22AssignVariableOp1assignvariableop_22_adam_basic_dqn_dense_3_bias_vIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_22n
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:2
Identity_23¼
AssignVariableOp_23AssignVariableOp4assignvariableop_23_adam_basic_dqn_q_values_kernel_vIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_23n
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:2
Identity_24ŗ
AssignVariableOp_24AssignVariableOp2assignvariableop_24_adam_basic_dqn_q_values_bias_vIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_249
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOp
Identity_25Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2
Identity_25÷
Identity_26IdentityIdentity_25:output:0^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*
T0*
_output_shapes
: 2
Identity_26"#
identity_26Identity_26:output:0*y
_input_shapesh
f: :::::::::::::::::::::::::2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112*
AssignVariableOp_12AssignVariableOp_122*
AssignVariableOp_13AssignVariableOp_132*
AssignVariableOp_14AssignVariableOp_142*
AssignVariableOp_15AssignVariableOp_152*
AssignVariableOp_16AssignVariableOp_162*
AssignVariableOp_17AssignVariableOp_172*
AssignVariableOp_18AssignVariableOp_182*
AssignVariableOp_19AssignVariableOp_192(
AssignVariableOp_2AssignVariableOp_22*
AssignVariableOp_20AssignVariableOp_202*
AssignVariableOp_21AssignVariableOp_212*
AssignVariableOp_22AssignVariableOp_222*
AssignVariableOp_23AssignVariableOp_232*
AssignVariableOp_24AssignVariableOp_242(
AssignVariableOp_3AssignVariableOp_32(
AssignVariableOp_4AssignVariableOp_42(
AssignVariableOp_5AssignVariableOp_52(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_9:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
«

 __inference__wrapped_model_88466
input_14
0basic_dqn_dense_2_matmul_readvariableop_resource5
1basic_dqn_dense_2_biasadd_readvariableop_resource4
0basic_dqn_dense_3_matmul_readvariableop_resource5
1basic_dqn_dense_3_biasadd_readvariableop_resource5
1basic_dqn_q_values_matmul_readvariableop_resource6
2basic_dqn_q_values_biasadd_readvariableop_resource
identityĆ
'basic_dqn/dense_2/MatMul/ReadVariableOpReadVariableOp0basic_dqn_dense_2_matmul_readvariableop_resource*
_output_shapes

: *
dtype02)
'basic_dqn/dense_2/MatMul/ReadVariableOpŖ
basic_dqn/dense_2/MatMulMatMulinput_1/basic_dqn/dense_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:’’’’’’’’’ 2
basic_dqn/dense_2/MatMulĀ
(basic_dqn/dense_2/BiasAdd/ReadVariableOpReadVariableOp1basic_dqn_dense_2_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02*
(basic_dqn/dense_2/BiasAdd/ReadVariableOpÉ
basic_dqn/dense_2/BiasAddBiasAdd"basic_dqn/dense_2/MatMul:product:00basic_dqn/dense_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:’’’’’’’’’ 2
basic_dqn/dense_2/BiasAdd
basic_dqn/dense_2/ReluRelu"basic_dqn/dense_2/BiasAdd:output:0*
T0*'
_output_shapes
:’’’’’’’’’ 2
basic_dqn/dense_2/ReluĆ
'basic_dqn/dense_3/MatMul/ReadVariableOpReadVariableOp0basic_dqn_dense_3_matmul_readvariableop_resource*
_output_shapes

:  *
dtype02)
'basic_dqn/dense_3/MatMul/ReadVariableOpĒ
basic_dqn/dense_3/MatMulMatMul$basic_dqn/dense_2/Relu:activations:0/basic_dqn/dense_3/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:’’’’’’’’’ 2
basic_dqn/dense_3/MatMulĀ
(basic_dqn/dense_3/BiasAdd/ReadVariableOpReadVariableOp1basic_dqn_dense_3_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02*
(basic_dqn/dense_3/BiasAdd/ReadVariableOpÉ
basic_dqn/dense_3/BiasAddBiasAdd"basic_dqn/dense_3/MatMul:product:00basic_dqn/dense_3/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:’’’’’’’’’ 2
basic_dqn/dense_3/BiasAdd
basic_dqn/dense_3/ReluRelu"basic_dqn/dense_3/BiasAdd:output:0*
T0*'
_output_shapes
:’’’’’’’’’ 2
basic_dqn/dense_3/ReluĘ
(basic_dqn/q_values/MatMul/ReadVariableOpReadVariableOp1basic_dqn_q_values_matmul_readvariableop_resource*
_output_shapes

: *
dtype02*
(basic_dqn/q_values/MatMul/ReadVariableOpŹ
basic_dqn/q_values/MatMulMatMul$basic_dqn/dense_3/Relu:activations:00basic_dqn/q_values/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:’’’’’’’’’2
basic_dqn/q_values/MatMulÅ
)basic_dqn/q_values/BiasAdd/ReadVariableOpReadVariableOp2basic_dqn_q_values_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02+
)basic_dqn/q_values/BiasAdd/ReadVariableOpĶ
basic_dqn/q_values/BiasAddBiasAdd#basic_dqn/q_values/MatMul:product:01basic_dqn/q_values/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:’’’’’’’’’2
basic_dqn/q_values/BiasAddw
IdentityIdentity#basic_dqn/q_values/BiasAdd:output:0*
T0*'
_output_shapes
:’’’’’’’’’2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:’’’’’’’’’:::::::P L
'
_output_shapes
:’’’’’’’’’
!
_user_specified_name	input_1
¹
µ
#__inference_signature_wrapper_88596
input_1
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:’’’’’’’’’*(
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8 *)
f$R"
 __inference__wrapped_model_884662
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:’’’’’’’’’2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:’’’’’’’’’::::::22
StatefulPartitionedCallStatefulPartitionedCall:P L
'
_output_shapes
:’’’’’’’’’
!
_user_specified_name	input_1
Ģ
«
C__inference_q_values_layer_call_and_return_conditional_losses_88646

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

: *
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:’’’’’’’’’2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:’’’’’’’’’2	
BiasAddd
IdentityIdentityBiasAdd:output:0*
T0*'
_output_shapes
:’’’’’’’’’2

Identity"
identityIdentity:output:0*.
_input_shapes
:’’’’’’’’’ :::O K
'
_output_shapes
:’’’’’’’’’ 
 
_user_specified_nameinputs
Ģ
«
C__inference_q_values_layer_call_and_return_conditional_losses_88534

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

: *
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:’’’’’’’’’2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:’’’’’’’’’2	
BiasAddd
IdentityIdentityBiasAdd:output:0*
T0*'
_output_shapes
:’’’’’’’’’2

Identity"
identityIdentity:output:0*.
_input_shapes
:’’’’’’’’’ :::O K
'
_output_shapes
:’’’’’’’’’ 
 
_user_specified_nameinputs
§;
į
__inference__traced_save_88753
file_prefix7
3savev2_basic_dqn_dense_2_kernel_read_readvariableop5
1savev2_basic_dqn_dense_2_bias_read_readvariableop7
3savev2_basic_dqn_dense_3_kernel_read_readvariableop5
1savev2_basic_dqn_dense_3_bias_read_readvariableop8
4savev2_basic_dqn_q_values_kernel_read_readvariableop6
2savev2_basic_dqn_q_values_bias_read_readvariableop(
$savev2_adam_iter_read_readvariableop	*
&savev2_adam_beta_1_read_readvariableop*
&savev2_adam_beta_2_read_readvariableop)
%savev2_adam_decay_read_readvariableop1
-savev2_adam_learning_rate_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop>
:savev2_adam_basic_dqn_dense_2_kernel_m_read_readvariableop<
8savev2_adam_basic_dqn_dense_2_bias_m_read_readvariableop>
:savev2_adam_basic_dqn_dense_3_kernel_m_read_readvariableop<
8savev2_adam_basic_dqn_dense_3_bias_m_read_readvariableop?
;savev2_adam_basic_dqn_q_values_kernel_m_read_readvariableop=
9savev2_adam_basic_dqn_q_values_bias_m_read_readvariableop>
:savev2_adam_basic_dqn_dense_2_kernel_v_read_readvariableop<
8savev2_adam_basic_dqn_dense_2_bias_v_read_readvariableop>
:savev2_adam_basic_dqn_dense_3_kernel_v_read_readvariableop<
8savev2_adam_basic_dqn_dense_3_bias_v_read_readvariableop?
;savev2_adam_basic_dqn_q_values_kernel_v_read_readvariableop=
9savev2_adam_basic_dqn_q_values_bias_v_read_readvariableop
savev2_const

identity_1¢MergeV2Checkpoints
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*2
StaticRegexFullMatchc
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.part2
Const
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*<
value3B1 B+_temp_93f69be8f1174849b60abb520a0fa036/part2	
Const_1
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: 2
Selectt

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: 2

StringJoinZ

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :2

num_shards
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : 2
ShardedFilename/shard¦
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilename
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*
valueBB%fc1/kernel/.ATTRIBUTES/VARIABLE_VALUEB#fc1/bias/.ATTRIBUTES/VARIABLE_VALUEB%fc2/kernel/.ATTRIBUTES/VARIABLE_VALUEB#fc2/bias/.ATTRIBUTES/VARIABLE_VALUEB(logits/kernel/.ATTRIBUTES/VARIABLE_VALUEB&logits/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEBAfc1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEB?fc1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBAfc2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEB?fc2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBDlogits/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBlogits/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBAfc1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB?fc1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBAfc2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB?fc2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBDlogits/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBlogits/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2/tensor_names¼
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*G
value>B<B B B B B B B B B B B B B B B B B B B B B B B B B B 2
SaveV2/shape_and_sliceså
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:03savev2_basic_dqn_dense_2_kernel_read_readvariableop1savev2_basic_dqn_dense_2_bias_read_readvariableop3savev2_basic_dqn_dense_3_kernel_read_readvariableop1savev2_basic_dqn_dense_3_bias_read_readvariableop4savev2_basic_dqn_q_values_kernel_read_readvariableop2savev2_basic_dqn_q_values_bias_read_readvariableop$savev2_adam_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableop-savev2_adam_learning_rate_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop:savev2_adam_basic_dqn_dense_2_kernel_m_read_readvariableop8savev2_adam_basic_dqn_dense_2_bias_m_read_readvariableop:savev2_adam_basic_dqn_dense_3_kernel_m_read_readvariableop8savev2_adam_basic_dqn_dense_3_bias_m_read_readvariableop;savev2_adam_basic_dqn_q_values_kernel_m_read_readvariableop9savev2_adam_basic_dqn_q_values_bias_m_read_readvariableop:savev2_adam_basic_dqn_dense_2_kernel_v_read_readvariableop8savev2_adam_basic_dqn_dense_2_bias_v_read_readvariableop:savev2_adam_basic_dqn_dense_3_kernel_v_read_readvariableop8savev2_adam_basic_dqn_dense_3_bias_v_read_readvariableop;savev2_adam_basic_dqn_q_values_kernel_v_read_readvariableop9savev2_adam_basic_dqn_q_values_bias_v_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *(
dtypes
2	2
SaveV2ŗ
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:2(
&MergeV2Checkpoints/checkpoint_prefixes”
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*
_output_shapes
 2
MergeV2Checkpointsr
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: 2

Identitym

Identity_1IdentityIdentity:output:0^MergeV2Checkpoints*
T0*
_output_shapes
: 2

Identity_1"!

identity_1Identity_1:output:0*·
_input_shapes„
¢: : : :  : : :: : : : : : : : : :  : : :: : :  : : :: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:$ 

_output_shapes

: : 

_output_shapes
: :$ 

_output_shapes

:  : 

_output_shapes
: :$ 

_output_shapes

: : 

_output_shapes
::

_output_shapes
: :

_output_shapes
: :	

_output_shapes
: :


_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :$ 

_output_shapes

: : 

_output_shapes
: :$ 

_output_shapes

:  : 

_output_shapes
: :$ 

_output_shapes

: : 

_output_shapes
::$ 

_output_shapes

: : 

_output_shapes
: :$ 

_output_shapes

:  : 

_output_shapes
: :$ 

_output_shapes

: : 

_output_shapes
::

_output_shapes
: 
§
Ŗ
B__inference_dense_3_layer_call_and_return_conditional_losses_88508

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:  *
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:’’’’’’’’’ 2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:’’’’’’’’’ 2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:’’’’’’’’’ 2
Reluf
IdentityIdentityRelu:activations:0*
T0*'
_output_shapes
:’’’’’’’’’ 2

Identity"
identityIdentity:output:0*.
_input_shapes
:’’’’’’’’’ :::O K
'
_output_shapes
:’’’’’’’’’ 
 
_user_specified_nameinputs
§
Ŗ
B__inference_dense_2_layer_call_and_return_conditional_losses_88481

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

: *
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:’’’’’’’’’ 2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:’’’’’’’’’ 2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:’’’’’’’’’ 2
Reluf
IdentityIdentityRelu:activations:0*
T0*'
_output_shapes
:’’’’’’’’’ 2

Identity"
identityIdentity:output:0*.
_input_shapes
:’’’’’’’’’:::O K
'
_output_shapes
:’’’’’’’’’
 
_user_specified_nameinputs
§
Ŗ
B__inference_dense_3_layer_call_and_return_conditional_losses_88627

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:  *
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:’’’’’’’’’ 2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:’’’’’’’’’ 2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:’’’’’’’’’ 2
Reluf
IdentityIdentityRelu:activations:0*
T0*'
_output_shapes
:’’’’’’’’’ 2

Identity"
identityIdentity:output:0*.
_input_shapes
:’’’’’’’’’ :::O K
'
_output_shapes
:’’’’’’’’’ 
 
_user_specified_nameinputs
§
Ŗ
B__inference_dense_2_layer_call_and_return_conditional_losses_88607

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

: *
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:’’’’’’’’’ 2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:’’’’’’’’’ 2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:’’’’’’’’’ 2
Reluf
IdentityIdentityRelu:activations:0*
T0*'
_output_shapes
:’’’’’’’’’ 2

Identity"
identityIdentity:output:0*.
_input_shapes
:’’’’’’’’’:::O K
'
_output_shapes
:’’’’’’’’’
 
_user_specified_nameinputs"øL
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*«
serving_default
;
input_10
serving_default_input_1:0’’’’’’’’’<
output_10
StatefulPartitionedCall:0’’’’’’’’’tensorflow/serving/predict:õZ
³
fc1
fc2

logits
	optimizer
	variables
regularization_losses
trainable_variables
	keras_api
	
signatures
F__call__
*G&call_and_return_all_conditional_losses
H_default_save_signature"Ź
_tf_keras_model°{"class_name": "Model", "name": "basic_dqn", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "config": {"layer was saved without config": true}, "is_graph_network": false, "keras_version": "2.4.0", "backend": "tensorflow", "model_config": {"class_name": "Model"}, "training_config": {"loss": "mse", "metrics": null, "weighted_metrics": null, "loss_weights": null, "optimizer_config": {"class_name": "Adam", "config": {"name": "Adam", "clipvalue": 10.0, "learning_rate": 0.001500000013038516, "decay": 0.0, "beta_1": 0.8999999761581421, "beta_2": 0.9990000128746033, "epsilon": 1e-07, "amsgrad": false}}}}
ź


kernel
bias
	variables
regularization_losses
trainable_variables
	keras_api
I__call__
*J&call_and_return_all_conditional_losses"Å
_tf_keras_layer«{"class_name": "Dense", "name": "dense_2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_2", "trainable": true, "dtype": "float32", "units": 32, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "HeUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 5}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 5]}}
ģ

kernel
bias
	variables
regularization_losses
trainable_variables
	keras_api
K__call__
*L&call_and_return_all_conditional_losses"Ē
_tf_keras_layer­{"class_name": "Dense", "name": "dense_3", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_3", "trainable": true, "dtype": "float32", "units": 32, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "HeUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 32}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 32]}}
ó

kernel
bias
	variables
regularization_losses
trainable_variables
	keras_api
M__call__
*N&call_and_return_all_conditional_losses"Ī
_tf_keras_layer“{"class_name": "Dense", "name": "q_values", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "q_values", "trainable": true, "dtype": "float32", "units": 3, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 32}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 32]}}
æ
iter

beta_1

beta_2
	decay
 learning_rate
m:m;m<m=m>m?
v@vAvBvCvDvE"
	optimizer
J

0
1
2
3
4
5"
trackable_list_wrapper
 "
trackable_list_wrapper
J

0
1
2
3
4
5"
trackable_list_wrapper
Ź

!layers
"metrics
	variables
#layer_metrics
$layer_regularization_losses
%non_trainable_variables
regularization_losses
trainable_variables
F__call__
H_default_save_signature
*G&call_and_return_all_conditional_losses
&G"call_and_return_conditional_losses"
_generic_user_object
,
Oserving_default"
signature_map
*:( 2basic_dqn/dense_2/kernel
$:" 2basic_dqn/dense_2/bias
.

0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
.

0
1"
trackable_list_wrapper
­

&layers
'metrics
(layer_metrics
	variables
)layer_regularization_losses
*non_trainable_variables
regularization_losses
trainable_variables
I__call__
*J&call_and_return_all_conditional_losses
&J"call_and_return_conditional_losses"
_generic_user_object
*:(  2basic_dqn/dense_3/kernel
$:" 2basic_dqn/dense_3/bias
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
­

+layers
,metrics
-layer_metrics
	variables
.layer_regularization_losses
/non_trainable_variables
regularization_losses
trainable_variables
K__call__
*L&call_and_return_all_conditional_losses
&L"call_and_return_conditional_losses"
_generic_user_object
+:) 2basic_dqn/q_values/kernel
%:#2basic_dqn/q_values/bias
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
­

0layers
1metrics
2layer_metrics
	variables
3layer_regularization_losses
4non_trainable_variables
regularization_losses
trainable_variables
M__call__
*N&call_and_return_all_conditional_losses
&N"call_and_return_conditional_losses"
_generic_user_object
:	 (2	Adam/iter
: (2Adam/beta_1
: (2Adam/beta_2
: (2
Adam/decay
: (2Adam/learning_rate
5
0
1
2"
trackable_list_wrapper
'
50"
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
»
	6total
	7count
8	variables
9	keras_api"
_tf_keras_metricj{"class_name": "Mean", "name": "loss", "dtype": "float32", "config": {"name": "loss", "dtype": "float32"}}
:  (2total
:  (2count
.
60
71"
trackable_list_wrapper
-
8	variables"
_generic_user_object
/:- 2Adam/basic_dqn/dense_2/kernel/m
):' 2Adam/basic_dqn/dense_2/bias/m
/:-  2Adam/basic_dqn/dense_3/kernel/m
):' 2Adam/basic_dqn/dense_3/bias/m
0:. 2 Adam/basic_dqn/q_values/kernel/m
*:(2Adam/basic_dqn/q_values/bias/m
/:- 2Adam/basic_dqn/dense_2/kernel/v
):' 2Adam/basic_dqn/dense_2/bias/v
/:-  2Adam/basic_dqn/dense_3/kernel/v
):' 2Adam/basic_dqn/dense_3/bias/v
0:. 2 Adam/basic_dqn/q_values/kernel/v
*:(2Adam/basic_dqn/q_values/bias/v
÷2ō
)__inference_basic_dqn_layer_call_fn_88569Ę
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŖ *&¢#
!
input_1’’’’’’’’’
2
D__inference_basic_dqn_layer_call_and_return_conditional_losses_88551Ę
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŖ *&¢#
!
input_1’’’’’’’’’
Ž2Ū
 __inference__wrapped_model_88466¶
²
FullArgSpec
args 
varargsjargs
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŖ *&¢#
!
input_1’’’’’’’’’
Ń2Ī
'__inference_dense_2_layer_call_fn_88616¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŖ *
 
ģ2é
B__inference_dense_2_layer_call_and_return_conditional_losses_88607¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŖ *
 
Ń2Ī
'__inference_dense_3_layer_call_fn_88636¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŖ *
 
ģ2é
B__inference_dense_3_layer_call_and_return_conditional_losses_88627¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŖ *
 
Ņ2Ļ
(__inference_q_values_layer_call_fn_88655¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŖ *
 
ķ2ź
C__inference_q_values_layer_call_and_return_conditional_losses_88646¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŖ *
 
2B0
#__inference_signature_wrapper_88596input_1
 __inference__wrapped_model_88466o
0¢-
&¢#
!
input_1’’’’’’’’’
Ŗ "3Ŗ0
.
output_1"
output_1’’’’’’’’’©
D__inference_basic_dqn_layer_call_and_return_conditional_losses_88551a
0¢-
&¢#
!
input_1’’’’’’’’’
Ŗ "%¢"

0’’’’’’’’’
 
)__inference_basic_dqn_layer_call_fn_88569T
0¢-
&¢#
!
input_1’’’’’’’’’
Ŗ "’’’’’’’’’¢
B__inference_dense_2_layer_call_and_return_conditional_losses_88607\
/¢,
%¢"
 
inputs’’’’’’’’’
Ŗ "%¢"

0’’’’’’’’’ 
 z
'__inference_dense_2_layer_call_fn_88616O
/¢,
%¢"
 
inputs’’’’’’’’’
Ŗ "’’’’’’’’’ ¢
B__inference_dense_3_layer_call_and_return_conditional_losses_88627\/¢,
%¢"
 
inputs’’’’’’’’’ 
Ŗ "%¢"

0’’’’’’’’’ 
 z
'__inference_dense_3_layer_call_fn_88636O/¢,
%¢"
 
inputs’’’’’’’’’ 
Ŗ "’’’’’’’’’ £
C__inference_q_values_layer_call_and_return_conditional_losses_88646\/¢,
%¢"
 
inputs’’’’’’’’’ 
Ŗ "%¢"

0’’’’’’’’’
 {
(__inference_q_values_layer_call_fn_88655O/¢,
%¢"
 
inputs’’’’’’’’’ 
Ŗ "’’’’’’’’’”
#__inference_signature_wrapper_88596z
;¢8
¢ 
1Ŗ.
,
input_1!
input_1’’’’’’’’’"3Ŗ0
.
output_1"
output_1’’’’’’’’’