from Model import *

# Testing Block
torch.manual_seed(0)

# Sample input
src_seq_len = 5
target_seq_len = 6
src_seq = torch.randint(
    0, 1000, (1, src_seq_len)
)  # Batch size 1, sequence length 10, random vocabulary indices
target_seq = torch.randint(
    0, 1500, (1, target_seq_len)
)  # Batch size 1, sequence length 10, random vocabulary indices
print(src_seq)
print(target_seq)
print("#######################################################################")


# vocab and model dim
src_vocab_size = 1000
target_vocab_size = 1500
d_model = 4

# Instantiate InputEmbeddings
src_input_embeddings = InputEmbeddings(src_vocab_size, d_model)
target_input_embeddings = InputEmbeddings(target_vocab_size, d_model)

# positional encodings for source and target
src_positional_encoding = PositionalEncoding(d_model, src_seq_len)
target_positional_encoding = PositionalEncoding(d_model, target_seq_len)

# Generate input embeddings
src_embeddings = src_input_embeddings(src_seq)
target_embeddings = target_input_embeddings(target_seq)

# Add positional encoding
src_encoded_input = src_positional_encoding(src_embeddings)
target_encoded_input = target_positional_encoding(target_embeddings)

# feed forward neural network configs
d_ff = 2048
dropout = 0.1

# Instantiate Encoder
encoder_layer = Encoder(d_model, d_ff, dropout)

# Pass encoded input through the encoder layer
encoder_output = encoder_layer(src_encoded_input)

# Print the output of encoder shape
print("Output shape encoder:", encoder_output.shape)
print(encoder_output)
print("#######################################################################")
# Instantiate decoder
decoder_layer = Decoder(d_model, d_ff, dropout)

# Pass encoded input through the decoder layer
decoder_output = decoder_layer(target_encoded_input, encoder_output)

# Print the output of decoder shape
print("Output shape decoder:", decoder_output.shape)
print(decoder_output)
print("#######################################################################")
# projection layer
projection_layer = ProjectionLayer(d_model, target_vocab_size)

# Pass decoder output through the projection layer
proj_output = projection_layer(decoder_output)

# Print the output shape
print("Output shape projection:", proj_output.shape)
print(proj_output)
print("#######################################################################")
# apply softmax on projected output
softmax_output = F.softmax(proj_output, dim=-1)
print(f"logits_output_shape : {softmax_output.shape}")
print(f"logits : {softmax_output}")
print("#######################################################################")
