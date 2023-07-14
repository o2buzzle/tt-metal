TT_DNN_SRCS = \
	libs/tt_dnn/op_library/auto_format.cpp \
	libs/tt_dnn/op_library/data_transfer/data_transfer_op.cpp \
	libs/tt_dnn/op_library/layout_conversion/layout_conversion_op.cpp \
	libs/tt_dnn/op_library/eltwise_binary/eltwise_binary_op.cpp \
	libs/tt_dnn/op_library/eltwise_binary/single_core/eltwise_binary_op_single_core.cpp \
	libs/tt_dnn/op_library/eltwise_binary/multi_core/eltwise_binary_op_multi_core.cpp \
	libs/tt_dnn/op_library/eltwise_unary/eltwise_unary_op.cpp \
	libs/tt_dnn/op_library/eltwise_unary/single_core/eltwise_unary_op_single_core.cpp \
	libs/tt_dnn/op_library/eltwise_unary/multi_core/eltwise_unary_op_multi_core.cpp \
	libs/tt_dnn/op_library/pad/pad_op.cpp \
	libs/tt_dnn/op_library/unpad/unpad_op.cpp \
	libs/tt_dnn/op_library/fill_rm/fill_rm_op.cpp \
	libs/tt_dnn/op_library/transpose/transpose_op.cpp \
	libs/tt_dnn/op_library/transpose/wh_multi_core/transpose_wh_op_multi_core.cpp \
	libs/tt_dnn/op_library/transpose/hc_multi_core/transpose_hc_op_multi_core.cpp \
	libs/tt_dnn/op_library/transpose/single_core/transpose_op_single_core.cpp \
	libs/tt_dnn/op_library/reduce/reduce_op.cpp \
	libs/tt_dnn/op_library/reduce/single_core/reduce_op_single_core.cpp \
	libs/tt_dnn/op_library/reduce/multi_core_h/reduce_op_multi_core_h.cpp \
	libs/tt_dnn/op_library/reduce/multi_core_w/reduce_op_multi_core_w.cpp \
	libs/tt_dnn/op_library/bcast/bcast_op.cpp \
	libs/tt_dnn/op_library/bcast/single_core/bcast_op_single_core.cpp \
	libs/tt_dnn/op_library/bcast/multi_core_h/bcast_op_multi_core_h.cpp \
	libs/tt_dnn/op_library/bcast/multi_core_w/bcast_op_multi_core_w.cpp \
	libs/tt_dnn/op_library/bcast/multi_core_hw/bcast_op_multi_core_hw.cpp \
	libs/tt_dnn/op_library/bmm/bmm_op.cpp \
	libs/tt_dnn/op_library/bmm/single_core/bmm_op_single_core.cpp \
	libs/tt_dnn/op_library/bmm/single_core/bmm_op_single_core_single_block.cpp \
	libs/tt_dnn/op_library/bmm/single_core/bmm_op_single_core_tilize_untilize.cpp \
	libs/tt_dnn/op_library/bmm/multi_core/bmm_op_multi_core.cpp \
	libs/tt_dnn/op_library/bmm/multi_core_reuse/bmm_op_multi_core_reuse.cpp \
	libs/tt_dnn/op_library/bmm/multi_core_reuse_mcast/bmm_op_multi_core_reuse_mcast.cpp \
	libs/tt_dnn/op_library/bmm/multi_core_reuse_generalized/bmm_op_multi_core_reuse_generalized.cpp \
	libs/tt_dnn/op_library/bmm/multi_core_reuse_mcast_generalized/bmm_op_multi_core_reuse_mcast_generalized.cpp \
	libs/tt_dnn/op_library/bmm/multi_core_reuse_padding/bmm_op_multi_core_reuse_padding.cpp \
	libs/tt_dnn/op_library/bmm/multi_core_reuse_mcast_padding/bmm_op_multi_core_reuse_mcast_padding.cpp \
	libs/tt_dnn/op_library/bmm/multi_core_reuse_mcast_optimized_bert_large/bmm_op_multi_core_reuse_mcast_optimized_bert_large.cpp \
	libs/tt_dnn/op_library/bmm/multi_core_reuse_optimized_bert_large/bmm_op_multi_core_reuse_optimized_bert_large.cpp \
	libs/tt_dnn/op_library/conv/conv_op.cpp \
	libs/tt_dnn/op_library/tilize/tilize_op.cpp \
	libs/tt_dnn/op_library/untilize/untilize_op.cpp \
	libs/tt_dnn/op_library/softmax/softmax_op.cpp \
	libs/tt_dnn/op_library/layernorm/layernorm_op.cpp \
	libs/tt_dnn/op_library/reshape/reshape_op.cpp \
	libs/tt_dnn/op_library/permute/permute_op.cpp \
	libs/tt_dnn/op_library/composite/composite_ops.cpp\
	libs/tt_dnn/op_library/bert_large_tms/bert_large_tms.cpp \
	libs/tt_dnn/op_library/bert_large_tms/multi_core_create_qkv_heads_from_fused_qkv/multi_core_create_qkv_heads_from_fused_qkv.cpp \
	libs/tt_dnn/op_library/bert_large_tms/multi_core_split_fused_qkv/multi_core_split_fused_qkv.cpp \
	libs/tt_dnn/op_library/bert_large_tms/multi_core_create_qkv_heads/multi_core_create_qkv_heads.cpp \
	libs/tt_dnn/op_library/bert_large_tms/multi_core_concat_heads/multi_core_concat_heads.cpp \
	libs/tt_dnn/op_library/run_operation.cpp \
	libs/tt_dnn/op_library/split/split_tiled.cpp \
	libs/tt_dnn/op_library/split/split_last_dim_two_chunks_tiled.cpp \
	libs/tt_dnn/op_library/operation_history.cpp \

TT_DNN_LIB = $(LIBDIR)/libtt_dnn.a
TT_DNN_DEFINES =
TT_DNN_INCLUDES = $(LIBS_INCLUDES)
TT_DNN_LDFLAGS = -lcommon -lllrt -ltt_metal -ltensor -ldtx
TT_DNN_CFLAGS = $(CFLAGS) -Werror -Wno-int-to-pointer-cast

TT_DNN_OBJS = $(addprefix $(OBJDIR)/, $(TT_DNN_SRCS:.cpp=.o))
TT_DNN_DEPS = $(addprefix $(OBJDIR)/, $(TT_DNN_SRCS:.cpp=.d))

-include $(TT_DNN_DEPS)

# Each module has a top level target as the entrypoint which must match the subdir name
libs/tt_dnn: $(TT_DNN_LIB)

$(TT_DNN_LIB): $(COMMON_LIB) $(DTX_LIB) $(TT_DNN_OBJS)
	@mkdir -p $(LIBDIR)
	ar rcs -o $@ $(TT_DNN_OBJS)

$(OBJDIR)/libs/tt_dnn/%.o: libs/tt_dnn/%.cpp
	@mkdir -p $(@D)
	$(CXX) $(TT_DNN_CFLAGS) $(CXXFLAGS) $(STATIC_LIB_FLAGS) $(TT_DNN_INCLUDES) $(TT_DNN_DEFINES) -c -o $@ $<
