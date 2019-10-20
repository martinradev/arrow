// Licensed to the Apache Software Foundation (ASF) under one
// or more contributor license agreements.  See the NOTICE file
// distributed with this work for additional information
// regarding copyright ownership.  The ASF licenses this file
// to you under the Apache License, Version 2.0 (the
// "License"); you may not use this file except in compliance
// with the License.  You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing,
// software distributed under the License is distributed on an
// "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, either express or implied.  See the License for the
// specific language governing permissions and limitations
// under the License.

#include "arrow/util/compression_zfp.h"

namespace arrow {
namespace util {

namespace {
Status CalculateNumberOfElements(zfp_type data_type, int64_t num_bytes, int64_t& num_elements) {
    int64_t data_type_size;
    switch(data_type) {
        case zfp_type_float:
            data_type_size = 4;
            break;
        case zfp_type_double:
            data_type_size = 8;
            break;
        default:
            return Status::TypeError("Unsupported type.");
    }
    num_elements = (num_bytes + data_type_size - 1) / data_type_size;
    return Status::OK();
}
} // namespace

ZFPCodec::ZFPCodec(zfp_type data_type, uint8_t precision)
 : data_type_(data_type),
   precision_(precision) {
}

Status ZFPCodec::Decompress(int64_t input_len, const uint8_t* input, int64_t output_buffer_len,
                            uint8_t* output_buffer) {
    int64_t output_len = 0;
    Status status = this->Decompress(input_len, input, output_buffer_len, output_buffer, &output_len);
    return status;
}

Status ZFPCodec::Decompress(int64_t input_len, const uint8_t* input, int64_t output_buffer_len,
                            uint8_t* output_buffer, int64_t* output_len) {
    bitstream *input_buffer_stream = stream_open(const_cast<uint8_t*>(input), input_len);
    zfp_stream *zfp = zfp_stream_open(input_buffer_stream);
    zfp_stream_rewind(zfp);

    zfp_field *zfp_field_ptr = zfp_field_alloc();
    const size_t num_bits_for_header = zfp_read_header(zfp, zfp_field_ptr, ZFP_HEADER_FULL);
    zfp_field_set_pointer(zfp_field_ptr, output_buffer);
    *output_len = zfp_field_size(zfp_field_ptr, NULLPTR);
    zfp_decompress(zfp, zfp_field_ptr);

    stream_close(input_buffer_stream);
    zfp_field_free(zfp_field_ptr);
    zfp_stream_close(zfp);

    return Status::OK();
}

Status ZFPCodec::Compress(int64_t input_len, const uint8_t* input, int64_t output_buffer_len,
                          uint8_t* output_buffer, int64_t* output_len) {
    int64_t num_elements = 0;
    Status status = CalculateNumberOfElements(data_type_, input_len, num_elements);
    if (!status.ok()) {
        return status;
    }
    // The const_cast here is necessary due to zfp's API design.
    zfp_field *zfp_field_ptr = zfp_field_1d(const_cast<uint8_t*>(input), data_type_, num_elements);
    zfp_stream *zfp = zfp_stream_open(NULL);
    zfp_stream_set_precision(zfp, precision_);
    bitstream* output_buffer_stream = stream_open(output_buffer, output_buffer_len);

    zfp_stream_set_bit_stream(zfp, output_buffer_stream);
    zfp_stream_rewind(zfp);

    const size_t num_bits_for_header = zfp_write_header(zfp, zfp_field_ptr, ZFP_HEADER_FULL);
    const size_t num_bytes_for_headers = (num_bits_for_header + size_t(7U)) / size_t(8U);
    const size_t compressed_data_size = zfp_compress(zfp, zfp_field_ptr);
    *output_len = num_bytes_for_headers + compressed_data_size;

    stream_close(output_buffer_stream);
    zfp_field_free(zfp_field_ptr);
    zfp_stream_close(zfp);

    return Status::OK();
}

int64_t ZFPCodec::MaxCompressedLen(int64_t input_len, const uint8_t* input) {
    int64_t num_elements = 0;
    Status status = CalculateNumberOfElements(data_type_, input_len, num_elements);
    (void)status;

    // The const_cast here is necessary due to zfp's API design.
    zfp_field *zfp_field_ptr = zfp_field_1d(const_cast<uint8_t*>(input), data_type_, num_elements);
    zfp_stream *zfp = zfp_stream_open(NULL);
    zfp_stream_set_precision(zfp, precision_);
    // HACK: add 128 extra bytes 
    const int64_t max_compressed_len = zfp_stream_maximum_size(zfp, zfp_field_ptr) + 128;

    zfp_field_free(zfp_field_ptr);
    zfp_stream_close(zfp);

    return max_compressed_len;
}

Status ZFPCodec::MakeCompressor(std::shared_ptr<Compressor>* out) {
    return Status::NotImplemented("");
}

Status ZFPCodec::MakeDecompressor(std::shared_ptr<Decompressor>* out) {
    return Status::NotImplemented("");
}

} // namespace util
} // namespace arrow
