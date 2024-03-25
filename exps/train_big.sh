tinygpt --name="MedGPT" \
--project="exps" \
--data_path="../dataset/shakespeare.txt" \
--device="cuda" \
--head_size 256 \
--num_head 8 \
--num_block 8 \
--chunk_size 256 \
--batch_size 64 \
