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

#include "parquet/encoding.h"

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <memory>
#include <utility>
#include <vector>

#include "arrow/array.h"
#include "arrow/builder.h"
#include "arrow/util/bit_stream_utils.h"
#include "arrow/util/checked_cast.h"
#include "arrow/util/hashing.h"
#include "arrow/util/logging.h"
#include "arrow/util/rle_encoding.h"
#include "arrow/util/ubsan.h"

#include "parquet/exception.h"
#include "parquet/platform.h"
#include "parquet/schema.h"
#include "parquet/types.h"

using arrow::Status;
using arrow::internal::checked_cast;

namespace parquet {

constexpr int64_t kInMemoryDefaultCapacity = 1024;

class EncoderImpl : virtual public Encoder {
 public:
  EncoderImpl(const ColumnDescriptor* descr, Encoding::type encoding, MemoryPool* pool)
      : descr_(descr),
        encoding_(encoding),
        pool_(pool),
        type_length_(descr ? descr->type_length() : -1) {}

  Encoding::type encoding() const override { return encoding_; }

  MemoryPool* memory_pool() const override { return pool_; }

 protected:
  // For accessing type-specific metadata, like FIXED_LEN_BYTE_ARRAY
  const ColumnDescriptor* descr_;
  const Encoding::type encoding_;
  MemoryPool* pool_;

  /// Type length from descr
  int type_length_;
};

// ----------------------------------------------------------------------
// Plain encoder implementation

template <typename DType>
class PlainEncoder : public EncoderImpl, virtual public TypedEncoder<DType> {
 public:
  using T = typename DType::c_type;

  explicit PlainEncoder(const ColumnDescriptor* descr, MemoryPool* pool)
      : EncoderImpl(descr, Encoding::PLAIN, pool), sink_(pool) {}

  int64_t EstimatedDataEncodedSize() override { return sink_.length(); }

  std::shared_ptr<Buffer> FlushValues() override {
    std::shared_ptr<Buffer> buffer;
    PARQUET_THROW_NOT_OK(sink_.Finish(&buffer));
    return buffer;
  }

  void Put(const T* buffer, int num_values) override;

  void Put(const arrow::Array& values) override;

  void PutSpaced(const T* src, int num_values, const uint8_t* valid_bits,
                 int64_t valid_bits_offset) override {
    std::shared_ptr<ResizableBuffer> buffer;
    PARQUET_THROW_NOT_OK(arrow::AllocateResizableBuffer(this->memory_pool(),
                                                        num_values * sizeof(T), &buffer));
    int32_t num_valid_values = 0;
    arrow::internal::BitmapReader valid_bits_reader(valid_bits, valid_bits_offset,
                                                    num_values);
    T* data = reinterpret_cast<T*>(buffer->mutable_data());
    for (int32_t i = 0; i < num_values; i++) {
      if (valid_bits_reader.IsSet()) {
        data[num_valid_values++] = src[i];
      }
      valid_bits_reader.Next();
    }
    Put(data, num_valid_values);
  }

  void UnsafePutByteArray(const void* data, uint32_t length) {
    DCHECK(length == 0 || data != nullptr) << "Value ptr cannot be NULL";
    sink_.UnsafeAppend(&length, sizeof(uint32_t));
    sink_.UnsafeAppend(data, static_cast<int64_t>(length));
  }

  void Put(const ByteArray& val) {
    // Write the result to the output stream
    const int64_t increment = static_cast<int64_t>(val.len + sizeof(uint32_t));
    if (ARROW_PREDICT_FALSE(sink_.length() + increment > sink_.capacity())) {
      PARQUET_THROW_NOT_OK(sink_.Reserve(increment));
    }
    UnsafePutByteArray(val.ptr, val.len);
  }

 protected:
  arrow::BufferBuilder sink_;
};

template <typename DType>
void PlainEncoder<DType>::Put(const T* buffer, int num_values) {
  if (num_values > 0) {
    PARQUET_THROW_NOT_OK(sink_.Append(buffer, num_values * sizeof(T)));
  }
}

template <>
inline void PlainEncoder<ByteArrayType>::Put(const ByteArray* src, int num_values) {
  for (int i = 0; i < num_values; ++i) {
    Put(src[i]);
  }
}

template <typename DType>
void PlainEncoder<DType>::Put(const arrow::Array& values) {
  ParquetException::NYI(values.type()->ToString());
}

void AssertBinary(const arrow::Array& values) {
  if (values.type_id() != arrow::Type::BINARY &&
      values.type_id() != arrow::Type::STRING) {
    throw ParquetException("Only BinaryArray and subclasses supported");
  }
}

template <>
void PlainEncoder<ByteArrayType>::Put(const arrow::Array& values) {
  AssertBinary(values);
  const auto& data = checked_cast<const arrow::BinaryArray&>(values);
  const int64_t total_bytes = data.value_offset(data.length()) - data.value_offset(0);
  PARQUET_THROW_NOT_OK(sink_.Reserve(total_bytes + data.length() * sizeof(uint32_t)));

  if (data.null_count() == 0) {
    // no nulls, just dump the data
    for (int64_t i = 0; i < data.length(); i++) {
      auto view = data.GetView(i);
      UnsafePutByteArray(view.data(), static_cast<uint32_t>(view.size()));
    }
  } else {
    for (int64_t i = 0; i < data.length(); i++) {
      if (data.IsValid(i)) {
        auto view = data.GetView(i);
        UnsafePutByteArray(view.data(), static_cast<uint32_t>(view.size()));
      }
    }
  }
}

template <>
inline void PlainEncoder<FLBAType>::Put(const FixedLenByteArray* src, int num_values) {
  if (descr_->type_length() == 0) {
    return;
  }
  for (int i = 0; i < num_values; ++i) {
    // Write the result to the output stream
    DCHECK(src[i].ptr != nullptr) << "Value ptr cannot be NULL";
    PARQUET_THROW_NOT_OK(sink_.Append(src[i].ptr, descr_->type_length()));
  }
}

class PlainFLBAEncoder : public PlainEncoder<FLBAType>, virtual public FLBAEncoder {
 public:
  using BASE = PlainEncoder<FLBAType>;
  using BASE::PlainEncoder;
};

class PlainBooleanEncoder : public EncoderImpl,
                            virtual public TypedEncoder<BooleanType>,
                            virtual public BooleanEncoder {
 public:
  explicit PlainBooleanEncoder(const ColumnDescriptor* descr, MemoryPool* pool)
      : EncoderImpl(descr, Encoding::PLAIN, pool),
        bits_available_(kInMemoryDefaultCapacity * 8),
        bits_buffer_(AllocateBuffer(pool, kInMemoryDefaultCapacity)),
        sink_(pool),
        bit_writer_(bits_buffer_->mutable_data(),
                    static_cast<int>(bits_buffer_->size())) {}

  int64_t EstimatedDataEncodedSize() override;
  std::shared_ptr<Buffer> FlushValues() override;

  void Put(const bool* src, int num_values) override;
  void Put(const std::vector<bool>& src, int num_values) override;

  void PutSpaced(const bool* src, int num_values, const uint8_t* valid_bits,
                 int64_t valid_bits_offset) override {
    std::shared_ptr<ResizableBuffer> buffer;
    PARQUET_THROW_NOT_OK(arrow::AllocateResizableBuffer(this->memory_pool(),
                                                        num_values * sizeof(T), &buffer));
    int32_t num_valid_values = 0;
    arrow::internal::BitmapReader valid_bits_reader(valid_bits, valid_bits_offset,
                                                    num_values);
    T* data = reinterpret_cast<T*>(buffer->mutable_data());
    for (int32_t i = 0; i < num_values; i++) {
      if (valid_bits_reader.IsSet()) {
        data[num_valid_values++] = src[i];
      }
      valid_bits_reader.Next();
    }
    Put(data, num_valid_values);
  }

  void Put(const arrow::Array& values) override {
    ParquetException::NYI("Direct Arrow to Boolean writes not implemented");
  }

 private:
  int bits_available_;
  std::shared_ptr<ResizableBuffer> bits_buffer_;
  arrow::BufferBuilder sink_;
  arrow::BitUtil::BitWriter bit_writer_;

  template <typename SequenceType>
  void PutImpl(const SequenceType& src, int num_values);
};

template <typename SequenceType>
void PlainBooleanEncoder::PutImpl(const SequenceType& src, int num_values) {
  int bit_offset = 0;
  if (bits_available_ > 0) {
    int bits_to_write = std::min(bits_available_, num_values);
    for (int i = 0; i < bits_to_write; i++) {
      bit_writer_.PutValue(src[i], 1);
    }
    bits_available_ -= bits_to_write;
    bit_offset = bits_to_write;

    if (bits_available_ == 0) {
      bit_writer_.Flush();
      PARQUET_THROW_NOT_OK(
          sink_.Append(bit_writer_.buffer(), bit_writer_.bytes_written()));
      bit_writer_.Clear();
    }
  }

  int bits_remaining = num_values - bit_offset;
  while (bit_offset < num_values) {
    bits_available_ = static_cast<int>(bits_buffer_->size()) * 8;

    int bits_to_write = std::min(bits_available_, bits_remaining);
    for (int i = bit_offset; i < bit_offset + bits_to_write; i++) {
      bit_writer_.PutValue(src[i], 1);
    }
    bit_offset += bits_to_write;
    bits_available_ -= bits_to_write;
    bits_remaining -= bits_to_write;

    if (bits_available_ == 0) {
      bit_writer_.Flush();
      PARQUET_THROW_NOT_OK(
          sink_.Append(bit_writer_.buffer(), bit_writer_.bytes_written()));
      bit_writer_.Clear();
    }
  }
}

int64_t PlainBooleanEncoder::EstimatedDataEncodedSize() {
  int64_t position = sink_.length();
  return position + bit_writer_.bytes_written();
}

std::shared_ptr<Buffer> PlainBooleanEncoder::FlushValues() {
  if (bits_available_ > 0) {
    bit_writer_.Flush();
    PARQUET_THROW_NOT_OK(sink_.Append(bit_writer_.buffer(), bit_writer_.bytes_written()));
    bit_writer_.Clear();
    bits_available_ = static_cast<int>(bits_buffer_->size()) * 8;
  }

  std::shared_ptr<Buffer> buffer;
  PARQUET_THROW_NOT_OK(sink_.Finish(&buffer));
  return buffer;
}

void PlainBooleanEncoder::Put(const bool* src, int num_values) {
  PutImpl(src, num_values);
}

void PlainBooleanEncoder::Put(const std::vector<bool>& src, int num_values) {
  PutImpl(src, num_values);
}

// ----------------------------------------------------------------------
// DictEncoder<T> implementations

template <typename DType>
struct DictEncoderTraits {
  using c_type = typename DType::c_type;
  using MemoTableType = arrow::internal::ScalarMemoTable<c_type>;
};

template <>
struct DictEncoderTraits<ByteArrayType> {
  using MemoTableType = arrow::internal::BinaryMemoTable;
};

template <>
struct DictEncoderTraits<FLBAType> {
  using MemoTableType = arrow::internal::BinaryMemoTable;
};

// Initially 1024 elements
static constexpr int32_t kInitialHashTableSize = 1 << 10;

/// See the dictionary encoding section of
/// https://github.com/Parquet/parquet-format.  The encoding supports
/// streaming encoding. Values are encoded as they are added while the
/// dictionary is being constructed. At any time, the buffered values
/// can be written out with the current dictionary size. More values
/// can then be added to the encoder, including new dictionary
/// entries.
template <typename DType>
class DictEncoderImpl : public EncoderImpl, virtual public DictEncoder<DType> {
  using MemoTableType = typename DictEncoderTraits<DType>::MemoTableType;

 public:
  typedef typename DType::c_type T;

  explicit DictEncoderImpl(const ColumnDescriptor* desc, MemoryPool* pool)
      : EncoderImpl(desc, Encoding::PLAIN_DICTIONARY, pool),
        dict_encoded_size_(0),
        memo_table_(pool, kInitialHashTableSize) {}

  ~DictEncoderImpl() override { DCHECK(buffered_indices_.empty()); }

  int dict_encoded_size() override { return dict_encoded_size_; }

  int WriteIndices(uint8_t* buffer, int buffer_len) override {
    // Write bit width in first byte
    *buffer = static_cast<uint8_t>(bit_width());
    ++buffer;
    --buffer_len;

    arrow::util::RleEncoder encoder(buffer, buffer_len, bit_width());

    for (int32_t index : buffered_indices_) {
      if (!encoder.Put(index)) return -1;
    }
    encoder.Flush();

    ClearIndices();
    return 1 + encoder.len();
  }

  void set_type_length(int type_length) { this->type_length_ = type_length; }

  /// Returns a conservative estimate of the number of bytes needed to encode the buffered
  /// indices. Used to size the buffer passed to WriteIndices().
  int64_t EstimatedDataEncodedSize() override {
    // Note: because of the way RleEncoder::CheckBufferFull() is called, we have to
    // reserve
    // an extra "RleEncoder::MinBufferSize" bytes. These extra bytes won't be used
    // but not reserving them would cause the encoder to fail.
    return 1 +
           arrow::util::RleEncoder::MaxBufferSize(
               bit_width(), static_cast<int>(buffered_indices_.size())) +
           arrow::util::RleEncoder::MinBufferSize(bit_width());
  }

  /// The minimum bit width required to encode the currently buffered indices.
  int bit_width() const override {
    if (ARROW_PREDICT_FALSE(num_entries() == 0)) return 0;
    if (ARROW_PREDICT_FALSE(num_entries() == 1)) return 1;
    return BitUtil::Log2(num_entries());
  }

  /// Encode value. Note that this does not actually write any data, just
  /// buffers the value's index to be written later.
  inline void Put(const T& value);

  // Not implemented for other data types
  inline void PutByteArray(const void* ptr, int32_t length);

  void Put(const T* src, int num_values) override {
    for (int32_t i = 0; i < num_values; i++) {
      Put(src[i]);
    }
  }

  void PutSpaced(const T* src, int num_values, const uint8_t* valid_bits,
                 int64_t valid_bits_offset) override {
    arrow::internal::BitmapReader valid_bits_reader(valid_bits, valid_bits_offset,
                                                    num_values);
    for (int32_t i = 0; i < num_values; i++) {
      if (valid_bits_reader.IsSet()) {
        Put(src[i]);
      }
      valid_bits_reader.Next();
    }
  }

  void Put(const arrow::Array& values) override;
  void PutDictionary(const arrow::Array& values) override;

  template <typename ArrowType>
  void PutIndicesTyped(const arrow::Array& data) {
    using ArrayType = typename arrow::TypeTraits<ArrowType>::ArrayType;
    const auto& indices = checked_cast<const ArrayType&>(data);
    auto values = indices.raw_values();

    size_t buffer_position = buffered_indices_.size();
    buffered_indices_.resize(
        buffer_position + static_cast<size_t>(indices.length() - indices.null_count()));
    if (indices.null_count() > 0) {
      arrow::internal::BitmapReader valid_bits_reader(indices.null_bitmap_data(),
                                                      indices.offset(), indices.length());
      for (int64_t i = 0; i < indices.length(); ++i) {
        if (valid_bits_reader.IsSet()) {
          buffered_indices_[buffer_position++] = static_cast<int32_t>(values[i]);
        }
        valid_bits_reader.Next();
      }
    } else {
      for (int64_t i = 0; i < indices.length(); ++i) {
        buffered_indices_[buffer_position++] = static_cast<int32_t>(values[i]);
      }
    }
  }

  void PutIndices(const arrow::Array& data) override {
    switch (data.type()->id()) {
      case arrow::Type::INT8:
        return PutIndicesTyped<arrow::Int8Type>(data);
      case arrow::Type::INT16:
        return PutIndicesTyped<arrow::Int16Type>(data);
      case arrow::Type::INT32:
        return PutIndicesTyped<arrow::Int32Type>(data);
      case arrow::Type::INT64:
        return PutIndicesTyped<arrow::Int64Type>(data);
      default:
        throw ParquetException("Dictionary indices were not signed integer");
    }
  }

  std::shared_ptr<Buffer> FlushValues() override {
    std::shared_ptr<ResizableBuffer> buffer =
        AllocateBuffer(this->pool_, EstimatedDataEncodedSize());
    int result_size = WriteIndices(buffer->mutable_data(),
                                   static_cast<int>(EstimatedDataEncodedSize()));
    PARQUET_THROW_NOT_OK(buffer->Resize(result_size, false));
    return std::move(buffer);
  }

  /// Writes out the encoded dictionary to buffer. buffer must be preallocated to
  /// dict_encoded_size() bytes.
  void WriteDict(uint8_t* buffer) override;

  /// The number of entries in the dictionary.
  int num_entries() const override { return memo_table_.size(); }

 private:
  /// Clears all the indices (but leaves the dictionary).
  void ClearIndices() { buffered_indices_.clear(); }

  /// Indices that have not yet be written out by WriteIndices().
  std::vector<int32_t> buffered_indices_;

  /// The number of bytes needed to encode the dictionary.
  int dict_encoded_size_;

  MemoTableType memo_table_;
};

template <typename DType>
void DictEncoderImpl<DType>::WriteDict(uint8_t* buffer) {
  // For primitive types, only a memcpy
  DCHECK_EQ(static_cast<size_t>(dict_encoded_size_), sizeof(T) * memo_table_.size());
  memo_table_.CopyValues(0 /* start_pos */, reinterpret_cast<T*>(buffer));
}

// ByteArray and FLBA already have the dictionary encoded in their data heaps
template <>
void DictEncoderImpl<ByteArrayType>::WriteDict(uint8_t* buffer) {
  memo_table_.VisitValues(0, [&buffer](const arrow::util::string_view& v) {
    uint32_t len = static_cast<uint32_t>(v.length());
    memcpy(buffer, &len, sizeof(len));
    buffer += sizeof(len);
    memcpy(buffer, v.data(), len);
    buffer += len;
  });
}

template <>
void DictEncoderImpl<FLBAType>::WriteDict(uint8_t* buffer) {
  memo_table_.VisitValues(0, [&](const arrow::util::string_view& v) {
    DCHECK_EQ(v.length(), static_cast<size_t>(type_length_));
    memcpy(buffer, v.data(), type_length_);
    buffer += type_length_;
  });
}

template <typename DType>
inline void DictEncoderImpl<DType>::Put(const T& v) {
  // Put() implementation for primitive types
  auto on_found = [](int32_t memo_index) {};
  auto on_not_found = [this](int32_t memo_index) {
    dict_encoded_size_ += static_cast<int>(sizeof(T));
  };

  auto memo_index = memo_table_.GetOrInsert(v, on_found, on_not_found);
  buffered_indices_.push_back(memo_index);
}

template <typename DType>
inline void DictEncoderImpl<DType>::PutByteArray(const void* ptr, int32_t length) {
  DCHECK(false);
}

template <>
inline void DictEncoderImpl<ByteArrayType>::PutByteArray(const void* ptr,
                                                         int32_t length) {
  static const uint8_t empty[] = {0};

  auto on_found = [](int32_t memo_index) {};
  auto on_not_found = [&](int32_t memo_index) {
    dict_encoded_size_ += static_cast<int>(length + sizeof(uint32_t));
  };

  DCHECK(ptr != nullptr || length == 0);
  ptr = (ptr != nullptr) ? ptr : empty;
  auto memo_index = memo_table_.GetOrInsert(ptr, length, on_found, on_not_found);
  buffered_indices_.push_back(memo_index);
}

template <>
inline void DictEncoderImpl<ByteArrayType>::Put(const ByteArray& val) {
  return PutByteArray(val.ptr, static_cast<int32_t>(val.len));
}

template <>
inline void DictEncoderImpl<FLBAType>::Put(const FixedLenByteArray& v) {
  static const uint8_t empty[] = {0};

  auto on_found = [](int32_t memo_index) {};
  auto on_not_found = [this](int32_t memo_index) { dict_encoded_size_ += type_length_; };

  DCHECK(v.ptr != nullptr || type_length_ == 0);
  const void* ptr = (v.ptr != nullptr) ? v.ptr : empty;
  auto memo_index = memo_table_.GetOrInsert(ptr, type_length_, on_found, on_not_found);
  buffered_indices_.push_back(memo_index);
}

template <typename DType>
void DictEncoderImpl<DType>::Put(const arrow::Array& values) {
  ParquetException::NYI(values.type()->ToString());
}

template <>
void DictEncoderImpl<ByteArrayType>::Put(const arrow::Array& values) {
  AssertBinary(values);
  const auto& data = checked_cast<const arrow::BinaryArray&>(values);
  if (data.null_count() == 0) {
    // no nulls, just dump the data
    for (int64_t i = 0; i < data.length(); i++) {
      auto view = data.GetView(i);
      PutByteArray(view.data(), static_cast<int32_t>(view.size()));
    }
  } else {
    for (int64_t i = 0; i < data.length(); i++) {
      if (data.IsValid(i)) {
        auto view = data.GetView(i);
        PutByteArray(view.data(), static_cast<int32_t>(view.size()));
      }
    }
  }
}

template <typename DType>
void DictEncoderImpl<DType>::PutDictionary(const arrow::Array& values) {
  ParquetException::NYI(values.type()->ToString());
}

template <>
void DictEncoderImpl<ByteArrayType>::PutDictionary(const arrow::Array& values) {
  AssertBinary(values);
  if (this->num_entries() > 0) {
    throw ParquetException("Can only call PutDictionary on an empty DictEncoder");
  }

  const auto& data = checked_cast<const arrow::BinaryArray&>(values);
  if (data.null_count() > 0) {
    throw ParquetException("Inserted binary dictionary cannot cannot contain nulls");
  }
  for (int64_t i = 0; i < data.length(); i++) {
    auto v = data.GetView(i);
    dict_encoded_size_ += static_cast<int>(v.size() + sizeof(uint32_t));
    ARROW_IGNORE_EXPR(
        memo_table_.GetOrInsert(v.data(), static_cast<int32_t>(v.size()),
                                /*on_found=*/[](int32_t memo_index) {},
                                /*on_not_found=*/[](int32_t memo_index) {}));
  }
}

// ----------------------------------------------------------------------
// FPByteStreamSplitEncoder<T> implementations

template <typename DType>
class FPByteStreamSplitEncoder : public EncoderImpl, virtual public TypedEncoder<DType> {
 public:
  using T = typename DType::c_type;

  explicit FPByteStreamSplitEncoder(const ColumnDescriptor* descr,
                        ::arrow::MemoryPool* pool = ::arrow::default_memory_pool());

  int64_t EstimatedDataEncodedSize() override;
  std::shared_ptr<Buffer> FlushValues() override;

  void Put(const T* buffer, int num_values) override;
  void Put(const ::arrow::Array& values) override;

  void PutSpaced(const T* src, int num_values, const uint8_t* valid_bits,
                         int64_t valid_bits_offset) override;

  int bit_width() const;
 protected:
  std::vector<T> values_;
};

template<typename DType>
FPByteStreamSplitEncoder<DType>::FPByteStreamSplitEncoder(
  const ColumnDescriptor* descr,
  ::arrow::MemoryPool* pool)
    : EncoderImpl(descr, Encoding::PLAIN, pool) {}

template<typename DType>
int64_t FPByteStreamSplitEncoder<DType>::EstimatedDataEncodedSize() {
    return
           arrow::util::RleEncoder::MaxBufferSize(
               bit_width(), static_cast<int>(values_.size() * sizeof(T))) +
           arrow::util::RleEncoder::MinBufferSize(bit_width());
}

template<typename DType>
int FPByteStreamSplitEncoder<DType>::bit_width() const {
    return 8;
}

template<typename DType>
std::shared_ptr<Buffer> FPByteStreamSplitEncoder<DType>::FlushValues() {
  constexpr size_t numStreams = sizeof(T);
  using UnsignedType = typename UnsignedTypeWithWidth<sizeof(T)>::type;
  std::shared_ptr<ResizableBuffer> buffer =
      AllocateBuffer(this->memory_pool(), EstimatedDataEncodedSize());
  uint8_t *mutableBuffer = buffer->mutable_data();
  arrow::util::RleEncoder encoder(mutableBuffer, buffer->size(), bit_width());
  for (size_t j = 0U; j < numStreams; ++j) {
    for (size_t i = 0; i < values_.size(); ++i) {
      const UnsignedType value = *reinterpret_cast<const UnsignedType*>(&values_[i]);
      const uint8_t byteInValue = static_cast<uint8_t>((value >> (8U*j)) & 0xFFU);
      encoder.Put(byteInValue);
    }
  }
  encoder.Flush();
  buffer->Resize(encoder.len(), false);
  values_.clear();
  return std::move(buffer);
}

template<typename DType>
void FPByteStreamSplitEncoder<DType>::Put(const ::arrow::Array& values) {
    throw ParquetException("Not supported");
}

template<typename DType>
void FPByteStreamSplitEncoder<DType>::Put(const T* buffer, int num_values) {
  values_.reserve(values_.size() + num_values);
  for (int i = 0; i < num_values; ++i) {
    values_.push_back(buffer[i]);
  }
}

template<typename DType>
void FPByteStreamSplitEncoder<DType>::PutSpaced(const T* src, int num_values, const uint8_t* valid_bits,
                        int64_t valid_bits_offset)
{
    throw ParquetException("Not supported");
}

// ----------------------------------------------------------------------
// Encoder and decoder factory functions

std::unique_ptr<Encoder> MakeEncoder(Type::type type_num, Encoding::type encoding,
                                     bool use_dictionary, const ColumnDescriptor* descr,
                                     MemoryPool* pool) {
  if (use_dictionary) {
    switch (type_num) {
      case Type::INT32:
        return std::unique_ptr<Encoder>(new DictEncoderImpl<Int32Type>(descr, pool));
      case Type::INT64:
        return std::unique_ptr<Encoder>(new DictEncoderImpl<Int64Type>(descr, pool));
      case Type::INT96:
        return std::unique_ptr<Encoder>(new DictEncoderImpl<Int96Type>(descr, pool));
      case Type::FLOAT:
        return std::unique_ptr<Encoder>(new DictEncoderImpl<FloatType>(descr, pool));
      case Type::DOUBLE:
        return std::unique_ptr<Encoder>(new DictEncoderImpl<DoubleType>(descr, pool));
      case Type::BYTE_ARRAY:
        return std::unique_ptr<Encoder>(new DictEncoderImpl<ByteArrayType>(descr, pool));
      case Type::FIXED_LEN_BYTE_ARRAY:
        return std::unique_ptr<Encoder>(new DictEncoderImpl<FLBAType>(descr, pool));
      default:
        DCHECK(false) << "Encoder not implemented";
        break;
    }
  } else if (encoding == Encoding::PLAIN) {
    switch (type_num) {
      case Type::BOOLEAN:
        return std::unique_ptr<Encoder>(new PlainBooleanEncoder(descr, pool));
      case Type::INT32:
        return std::unique_ptr<Encoder>(new PlainEncoder<Int32Type>(descr, pool));
      case Type::INT64:
        return std::unique_ptr<Encoder>(new PlainEncoder<Int64Type>(descr, pool));
      case Type::INT96:
        return std::unique_ptr<Encoder>(new PlainEncoder<Int96Type>(descr, pool));
      case Type::FLOAT:
        return std::unique_ptr<Encoder>(new PlainEncoder<FloatType>(descr, pool));
      case Type::DOUBLE:
        return std::unique_ptr<Encoder>(new PlainEncoder<DoubleType>(descr, pool));
      case Type::BYTE_ARRAY:
        return std::unique_ptr<Encoder>(new PlainEncoder<ByteArrayType>(descr, pool));
      case Type::FIXED_LEN_BYTE_ARRAY:
        return std::unique_ptr<Encoder>(new PlainEncoder<FLBAType>(descr, pool));
      default:
        DCHECK(false) << "Encoder not implemented";
        break;
    }
  } else if (encoding == Encoding::BYTE_STREAM_SPLIT) {
    switch (type_num) {
      case Type::FLOAT:
        return std::unique_ptr<Encoder>(new FPByteStreamSplitEncoder<FloatType>(descr, pool));
      case Type::DOUBLE:
        return std::unique_ptr<Encoder>(new FPByteStreamSplitEncoder<DoubleType>(descr, pool));
      default:
        DCHECK(false) << "Encoder not implemented";
        break;
    }
  } else {
    ParquetException::NYI("Selected encoding is not supported");
  }
  DCHECK(false) << "Should not be able to reach this code";
  return nullptr;
}

class DecoderImpl : virtual public Decoder {
 public:
  void SetData(int num_values, const uint8_t* data, int len) override {
    num_values_ = num_values;
    data_ = data;
    len_ = len;
  }

  int values_left() const override { return num_values_; }
  Encoding::type encoding() const override { return encoding_; }

 protected:
  explicit DecoderImpl(const ColumnDescriptor* descr, Encoding::type encoding)
      : descr_(descr), encoding_(encoding), num_values_(0), data_(NULLPTR), len_(0) {}

  // For accessing type-specific metadata, like FIXED_LEN_BYTE_ARRAY
  const ColumnDescriptor* descr_;

  const Encoding::type encoding_;
  int num_values_;
  const uint8_t* data_;
  int len_;
  int type_length_;
};

template <typename DType>
class PlainDecoder : public DecoderImpl, virtual public TypedDecoder<DType> {
 public:
  using T = typename DType::c_type;
  explicit PlainDecoder(const ColumnDescriptor* descr);

  int Decode(T* buffer, int max_values) override;
};

template <typename DType>
PlainDecoder<DType>::PlainDecoder(const ColumnDescriptor* descr)
    : DecoderImpl(descr, Encoding::PLAIN) {
  if (descr_ && descr_->physical_type() == Type::FIXED_LEN_BYTE_ARRAY) {
    type_length_ = descr_->type_length();
  } else {
    type_length_ = -1;
  }
}

// Decode routine templated on C++ type rather than type enum
template <typename T>
inline int DecodePlain(const uint8_t* data, int64_t data_size, int num_values,
                       int type_length, T* out) {
  int bytes_to_decode = num_values * static_cast<int>(sizeof(T));
  if (data_size < bytes_to_decode) {
    ParquetException::EofException();
  }
  // If bytes_to_decode == 0, data could be null
  if (bytes_to_decode > 0) {
    memcpy(out, data, bytes_to_decode);
  }
  return bytes_to_decode;
}

// Template specialization for BYTE_ARRAY. The written values do not own their
// own data.
template <>
inline int DecodePlain<ByteArray>(const uint8_t* data, int64_t data_size, int num_values,
                                  int type_length, ByteArray* out) {
  int bytes_decoded = 0;
  int increment;
  for (int i = 0; i < num_values; ++i) {
    uint32_t len = out[i].len = arrow::util::SafeLoadAs<uint32_t>(data);
    increment = static_cast<int>(sizeof(uint32_t) + len);
    if (data_size < increment) ParquetException::EofException();
    out[i].ptr = data + sizeof(uint32_t);
    data += increment;
    data_size -= increment;
    bytes_decoded += increment;
  }
  return bytes_decoded;
}

// Template specialization for FIXED_LEN_BYTE_ARRAY. The written values do not
// own their own data.
template <>
inline int DecodePlain<FixedLenByteArray>(const uint8_t* data, int64_t data_size,
                                          int num_values, int type_length,
                                          FixedLenByteArray* out) {
  int bytes_to_decode = type_length * num_values;
  if (data_size < bytes_to_decode) {
    ParquetException::EofException();
  }
  for (int i = 0; i < num_values; ++i) {
    out[i].ptr = data;
    data += type_length;
    data_size -= type_length;
  }
  return bytes_to_decode;
}

template <typename DType>
int PlainDecoder<DType>::Decode(T* buffer, int max_values) {
  max_values = std::min(max_values, num_values_);
  int bytes_consumed = DecodePlain<T>(data_, len_, max_values, type_length_, buffer);
  data_ += bytes_consumed;
  len_ -= bytes_consumed;
  num_values_ -= max_values;
  return max_values;
}

class PlainBooleanDecoder : public DecoderImpl,
                            virtual public TypedDecoder<BooleanType>,
                            virtual public BooleanDecoder {
 public:
  explicit PlainBooleanDecoder(const ColumnDescriptor* descr);
  void SetData(int num_values, const uint8_t* data, int len) override;

  // Two flavors of bool decoding
  int Decode(uint8_t* buffer, int max_values) override;
  int Decode(bool* buffer, int max_values) override;

 private:
  std::unique_ptr<arrow::BitUtil::BitReader> bit_reader_;
};

PlainBooleanDecoder::PlainBooleanDecoder(const ColumnDescriptor* descr)
    : DecoderImpl(descr, Encoding::PLAIN) {}

void PlainBooleanDecoder::SetData(int num_values, const uint8_t* data, int len) {
  num_values_ = num_values;
  bit_reader_.reset(new BitUtil::BitReader(data, len));
}

int PlainBooleanDecoder::Decode(uint8_t* buffer, int max_values) {
  max_values = std::min(max_values, num_values_);
  bool val;
  arrow::internal::BitmapWriter bit_writer(buffer, 0, max_values);
  for (int i = 0; i < max_values; ++i) {
    if (!bit_reader_->GetValue(1, &val)) {
      ParquetException::EofException();
    }
    if (val) {
      bit_writer.Set();
    }
    bit_writer.Next();
  }
  bit_writer.Finish();
  num_values_ -= max_values;
  return max_values;
}

int PlainBooleanDecoder::Decode(bool* buffer, int max_values) {
  max_values = std::min(max_values, num_values_);
  if (bit_reader_->GetBatch(1, buffer, max_values) != max_values) {
    ParquetException::EofException();
  }
  num_values_ -= max_values;
  return max_values;
}

struct ArrowBinaryHelper {
  explicit ArrowBinaryHelper(ArrowBinaryAccumulator* out) {
    this->out = out;
    this->builder = out->builder.get();
    this->chunk_space_remaining =
        ::arrow::kBinaryMemoryLimit - this->builder->value_data_length();
  }

  Status PushChunk() {
    std::shared_ptr<::arrow::Array> result;
    RETURN_NOT_OK(builder->Finish(&result));
    out->chunks.push_back(result);
    chunk_space_remaining = ::arrow::kBinaryMemoryLimit;
    return Status::OK();
  }

  bool CanFit(int64_t length) const { return length <= chunk_space_remaining; }

  void UnsafeAppend(const uint8_t* data, int32_t length) {
    chunk_space_remaining -= length;
    builder->UnsafeAppend(data, length);
  }

  void UnsafeAppendNull() { builder->UnsafeAppendNull(); }

  Status Append(const uint8_t* data, int32_t length) {
    chunk_space_remaining -= length;
    return builder->Append(data, length);
  }

  Status AppendNull() { return builder->AppendNull(); }

  ArrowBinaryAccumulator* out;
  arrow::BinaryBuilder* builder;
  int64_t chunk_space_remaining;
};

class PlainByteArrayDecoder : public PlainDecoder<ByteArrayType>,
                              virtual public ByteArrayDecoder {
 public:
  using Base = PlainDecoder<ByteArrayType>;
  using Base::DecodeSpaced;
  using Base::PlainDecoder;

  // ----------------------------------------------------------------------
  // Dictionary read paths

  int DecodeArrow(int num_values, int null_count, const uint8_t* valid_bits,
                  int64_t valid_bits_offset,
                  arrow::BinaryDictionary32Builder* builder) override {
    int result = 0;
    PARQUET_THROW_NOT_OK(DecodeArrow(num_values, null_count, valid_bits,
                                     valid_bits_offset, builder, &result));
    return result;
  }

  int DecodeArrowNonNull(int num_values,
                         arrow::BinaryDictionary32Builder* builder) override {
    int result = 0;
    PARQUET_THROW_NOT_OK(DecodeArrowNonNull(num_values, builder, &result));
    return result;
  }

  // ----------------------------------------------------------------------
  // Optimized dense binary read paths

  int DecodeArrow(int num_values, int null_count, const uint8_t* valid_bits,
                  int64_t valid_bits_offset, ArrowBinaryAccumulator* out) override {
    int result = 0;
    PARQUET_THROW_NOT_OK(DecodeArrowDense(num_values, null_count, valid_bits,
                                          valid_bits_offset, out, &result));
    return result;
  }

  int DecodeArrowNonNull(int num_values, ArrowBinaryAccumulator* out) override {
    int result = 0;
    PARQUET_THROW_NOT_OK(DecodeArrowDenseNonNull(num_values, out, &result));
    return result;
  }

 private:
  Status DecodeArrowDense(int num_values, int null_count, const uint8_t* valid_bits,
                          int64_t valid_bits_offset, ArrowBinaryAccumulator* out,
                          int* out_values_decoded) {
    ArrowBinaryHelper helper(out);
    arrow::internal::BitmapReader bit_reader(valid_bits, valid_bits_offset, num_values);
    int values_decoded = 0;

    RETURN_NOT_OK(helper.builder->Reserve(num_values));
    RETURN_NOT_OK(helper.builder->ReserveData(
        std::min<int64_t>(len_, helper.chunk_space_remaining)));
    for (int i = 0; i < num_values; ++i) {
      if (bit_reader.IsSet()) {
        auto value_len = static_cast<int32_t>(arrow::util::SafeLoadAs<uint32_t>(data_));
        int increment = static_cast<int>(sizeof(uint32_t) + value_len);
        if (ARROW_PREDICT_FALSE(len_ < increment)) ParquetException::EofException();
        if (ARROW_PREDICT_FALSE(!helper.CanFit(value_len))) {
          // This element would exceed the capacity of a chunk
          RETURN_NOT_OK(helper.PushChunk());
          RETURN_NOT_OK(helper.builder->Reserve(num_values - i));
          RETURN_NOT_OK(helper.builder->ReserveData(
              std::min<int64_t>(len_, helper.chunk_space_remaining)));
        }
        helper.UnsafeAppend(data_ + sizeof(uint32_t), value_len);
        data_ += increment;
        len_ -= increment;
        ++values_decoded;
      } else {
        helper.UnsafeAppendNull();
      }
      bit_reader.Next();
    }

    num_values_ -= values_decoded;
    *out_values_decoded = values_decoded;
    return Status::OK();
  }

  Status DecodeArrowDenseNonNull(int num_values, ArrowBinaryAccumulator* out,
                                 int* values_decoded) {
    ArrowBinaryHelper helper(out);
    num_values = std::min(num_values, num_values_);
    RETURN_NOT_OK(helper.builder->Reserve(num_values));
    RETURN_NOT_OK(helper.builder->ReserveData(
        std::min<int64_t>(len_, helper.chunk_space_remaining)));
    for (int i = 0; i < num_values; ++i) {
      int32_t value_len = static_cast<int32_t>(arrow::util::SafeLoadAs<uint32_t>(data_));
      int increment = static_cast<int>(sizeof(uint32_t) + value_len);
      if (ARROW_PREDICT_FALSE(len_ < increment)) ParquetException::EofException();
      if (ARROW_PREDICT_FALSE(!helper.CanFit(value_len))) {
        // This element would exceed the capacity of a chunk
        RETURN_NOT_OK(helper.PushChunk());
        RETURN_NOT_OK(helper.builder->Reserve(num_values - i));
        RETURN_NOT_OK(helper.builder->ReserveData(
            std::min<int64_t>(len_, helper.chunk_space_remaining)));
      }
      helper.UnsafeAppend(data_ + sizeof(uint32_t), value_len);
      data_ += increment;
      len_ -= increment;
    }

    num_values_ -= num_values;
    *values_decoded = num_values;
    return Status::OK();
  }

  template <typename BuilderType>
  Status DecodeArrow(int num_values, int null_count, const uint8_t* valid_bits,
                     int64_t valid_bits_offset, BuilderType* builder,
                     int* out_values_decoded) {
    RETURN_NOT_OK(builder->Reserve(num_values));
    arrow::internal::BitmapReader bit_reader(valid_bits, valid_bits_offset, num_values);
    int values_decoded = 0;
    for (int i = 0; i < num_values; ++i) {
      if (bit_reader.IsSet()) {
        uint32_t value_len = arrow::util::SafeLoadAs<uint32_t>(data_);
        int increment = static_cast<int>(sizeof(uint32_t) + value_len);
        if (len_ < increment) {
          ParquetException::EofException();
        }
        RETURN_NOT_OK(builder->Append(data_ + sizeof(uint32_t), value_len));
        data_ += increment;
        len_ -= increment;
        ++values_decoded;
      } else {
        RETURN_NOT_OK(builder->AppendNull());
      }
      bit_reader.Next();
    }
    num_values_ -= values_decoded;
    *out_values_decoded = values_decoded;
    return Status::OK();
  }

  template <typename BuilderType>
  Status DecodeArrowNonNull(int num_values, BuilderType* builder, int* values_decoded) {
    num_values = std::min(num_values, num_values_);
    RETURN_NOT_OK(builder->Reserve(num_values));
    for (int i = 0; i < num_values; ++i) {
      uint32_t value_len = arrow::util::SafeLoadAs<uint32_t>(data_);
      int increment = static_cast<int>(sizeof(uint32_t) + value_len);
      if (len_ < increment) ParquetException::EofException();
      RETURN_NOT_OK(builder->Append(data_ + sizeof(uint32_t), value_len));
      data_ += increment;
      len_ -= increment;
    }
    num_values_ -= num_values;
    *values_decoded = num_values;
    return Status::OK();
  }
};

class PlainFLBADecoder : public PlainDecoder<FLBAType>, virtual public FLBADecoder {
 public:
  using Base = PlainDecoder<FLBAType>;
  using Base::PlainDecoder;
};

// ----------------------------------------------------------------------
// Dictionary encoding and decoding

template <typename Type>
class DictDecoderImpl : public DecoderImpl, virtual public DictDecoder<Type> {
 public:
  typedef typename Type::c_type T;

  // Initializes the dictionary with values from 'dictionary'. The data in
  // dictionary is not guaranteed to persist in memory after this call so the
  // dictionary decoder needs to copy the data out if necessary.
  explicit DictDecoderImpl(const ColumnDescriptor* descr,
                           MemoryPool* pool = arrow::default_memory_pool())
      : DecoderImpl(descr, Encoding::RLE_DICTIONARY),
        dictionary_(AllocateBuffer(pool, 0)),
        dictionary_length_(0),
        byte_array_data_(AllocateBuffer(pool, 0)),
        byte_array_offsets_(AllocateBuffer(pool, 0)),
        indices_scratch_space_(AllocateBuffer(pool, 0)) {}

  // Perform type-specific initiatialization
  void SetDict(TypedDecoder<Type>* dictionary) override;

  void SetData(int num_values, const uint8_t* data, int len) override {
    num_values_ = num_values;
    if (len == 0) return;
    uint8_t bit_width = *data;
    ++data;
    --len;
    idx_decoder_ = arrow::util::RleDecoder(data, len, bit_width);
  }

  int Decode(T* buffer, int num_values) override {
    num_values = std::min(num_values, num_values_);
    int decoded_values = idx_decoder_.GetBatchWithDict(
        reinterpret_cast<const T*>(dictionary_->data()), buffer, num_values);
    if (decoded_values != num_values) {
      ParquetException::EofException();
    }
    num_values_ -= num_values;
    return num_values;
  }

  int DecodeSpaced(T* buffer, int num_values, int null_count, const uint8_t* valid_bits,
                   int64_t valid_bits_offset) override {
    num_values = std::min(num_values, num_values_);
    if (num_values != idx_decoder_.GetBatchWithDictSpaced(
                          reinterpret_cast<const T*>(dictionary_->data()), buffer,
                          num_values, null_count, valid_bits, valid_bits_offset)) {
      ParquetException::EofException();
    }
    num_values_ -= num_values;
    return num_values;
  }

  void InsertDictionary(arrow::ArrayBuilder* builder) override;

  int DecodeIndicesSpaced(int num_values, int null_count, const uint8_t* valid_bits,
                          int64_t valid_bits_offset,
                          arrow::ArrayBuilder* builder) override {
    if (num_values > 0) {
      // TODO(wesm): Refactor to batch reads for improved memory use. It is not
      // trivial because the null_count is relative to the entire bitmap
      PARQUET_THROW_NOT_OK(indices_scratch_space_->TypedResize<int32_t>(
          num_values, /*shrink_to_fit=*/false));
    }

    auto indices_buffer =
        reinterpret_cast<int32_t*>(indices_scratch_space_->mutable_data());

    if (num_values != idx_decoder_.GetBatchSpaced(num_values, null_count, valid_bits,
                                                  valid_bits_offset, indices_buffer)) {
      ParquetException::EofException();
    }

    /// XXX(wesm): Cannot append "valid bits" directly to the builder
    std::vector<uint8_t> valid_bytes(num_values);
    arrow::internal::BitmapReader bit_reader(valid_bits, valid_bits_offset, num_values);
    for (int64_t i = 0; i < num_values; ++i) {
      valid_bytes[i] = static_cast<uint8_t>(bit_reader.IsSet());
      bit_reader.Next();
    }

    auto binary_builder = checked_cast<arrow::BinaryDictionary32Builder*>(builder);
    PARQUET_THROW_NOT_OK(
        binary_builder->AppendIndices(indices_buffer, num_values, valid_bytes.data()));
    num_values_ -= num_values - null_count;
    return num_values - null_count;
  }

  int DecodeIndices(int num_values, arrow::ArrayBuilder* builder) override {
    num_values = std::min(num_values, num_values_);
    num_values = std::min(num_values, num_values_);
    if (num_values > 0) {
      // TODO(wesm): Refactor to batch reads for improved memory use. This is
      // relatively simple here because we don't have to do any bookkeeping of
      // nulls
      PARQUET_THROW_NOT_OK(indices_scratch_space_->TypedResize<int32_t>(
          num_values, /*shrink_to_fit=*/false));
    }
    auto indices_buffer =
        reinterpret_cast<int32_t*>(indices_scratch_space_->mutable_data());
    if (num_values != idx_decoder_.GetBatch(indices_buffer, num_values)) {
      ParquetException::EofException();
    }
    auto binary_builder = checked_cast<arrow::BinaryDictionary32Builder*>(builder);
    PARQUET_THROW_NOT_OK(binary_builder->AppendIndices(indices_buffer, num_values));
    num_values_ -= num_values;
    return num_values;
  }

 protected:
  inline void DecodeDict(TypedDecoder<Type>* dictionary) {
    dictionary_length_ = static_cast<int32_t>(dictionary->values_left());
    PARQUET_THROW_NOT_OK(dictionary_->Resize(dictionary_length_ * sizeof(T),
                                             /*shrink_to_fit=*/false));
    dictionary->Decode(reinterpret_cast<T*>(dictionary_->mutable_data()),
                       dictionary_length_);
  }

  // Only one is set.
  std::shared_ptr<ResizableBuffer> dictionary_;

  int32_t dictionary_length_;

  // Data that contains the byte array data (byte_array_dictionary_ just has the
  // pointers).
  std::shared_ptr<ResizableBuffer> byte_array_data_;

  // Arrow-style byte offsets for each dictionary value. We maintain two
  // representations of the dictionary, one as ByteArray* for non-Arrow
  // consumers and this one for Arrow conumers. Since dictionaries are
  // generally pretty small to begin with this doesn't mean too much extra
  // memory use in most cases
  std::shared_ptr<ResizableBuffer> byte_array_offsets_;

  // Reusable buffer for decoding dictionary indices to be appended to a
  // BinaryDictionary32Builder
  std::shared_ptr<ResizableBuffer> indices_scratch_space_;

  arrow::util::RleDecoder idx_decoder_;
};

template <typename Type>
void DictDecoderImpl<Type>::SetDict(TypedDecoder<Type>* dictionary) {
  DecodeDict(dictionary);
}

template <>
void DictDecoderImpl<BooleanType>::SetDict(TypedDecoder<BooleanType>* dictionary) {
  ParquetException::NYI("Dictionary encoding is not implemented for boolean values");
}

template <>
void DictDecoderImpl<ByteArrayType>::SetDict(TypedDecoder<ByteArrayType>* dictionary) {
  DecodeDict(dictionary);

  auto dict_values = reinterpret_cast<ByteArray*>(dictionary_->mutable_data());

  int total_size = 0;
  for (int i = 0; i < dictionary_length_; ++i) {
    total_size += dict_values[i].len;
  }
  if (total_size > 0) {
    PARQUET_THROW_NOT_OK(byte_array_data_->Resize(total_size,
                                                  /*shrink_to_fit=*/false));
    PARQUET_THROW_NOT_OK(
        byte_array_offsets_->Resize((dictionary_length_ + 1) * sizeof(int32_t),
                                    /*shrink_to_fit=*/false));
  }

  int32_t offset = 0;
  uint8_t* bytes_data = byte_array_data_->mutable_data();
  int32_t* bytes_offsets =
      reinterpret_cast<int32_t*>(byte_array_offsets_->mutable_data());
  for (int i = 0; i < dictionary_length_; ++i) {
    memcpy(bytes_data + offset, dict_values[i].ptr, dict_values[i].len);
    bytes_offsets[i] = offset;
    dict_values[i].ptr = bytes_data + offset;
    offset += dict_values[i].len;
  }
  bytes_offsets[dictionary_length_] = offset;
}

template <>
inline void DictDecoderImpl<FLBAType>::SetDict(TypedDecoder<FLBAType>* dictionary) {
  DecodeDict(dictionary);

  auto dict_values = reinterpret_cast<FLBA*>(dictionary_->mutable_data());

  int fixed_len = descr_->type_length();
  int total_size = dictionary_length_ * fixed_len;

  PARQUET_THROW_NOT_OK(byte_array_data_->Resize(total_size,
                                                /*shrink_to_fit=*/false));
  uint8_t* bytes_data = byte_array_data_->mutable_data();
  for (int32_t i = 0, offset = 0; i < dictionary_length_; ++i, offset += fixed_len) {
    memcpy(bytes_data + offset, dict_values[i].ptr, fixed_len);
    dict_values[i].ptr = bytes_data + offset;
  }
}

template <typename Type>
void DictDecoderImpl<Type>::InsertDictionary(arrow::ArrayBuilder* builder) {
  ParquetException::NYI("InsertDictionary only implemented for BYTE_ARRAY types");
}

template <>
void DictDecoderImpl<ByteArrayType>::InsertDictionary(arrow::ArrayBuilder* builder) {
  auto binary_builder = checked_cast<arrow::BinaryDictionary32Builder*>(builder);

  // Make an BinaryArray referencing the internal dictionary data
  auto arr = std::make_shared<arrow::BinaryArray>(dictionary_length_, byte_array_offsets_,
                                                  byte_array_data_);
  PARQUET_THROW_NOT_OK(binary_builder->InsertMemoValues(*arr));
}

class DictByteArrayDecoderImpl : public DictDecoderImpl<ByteArrayType>,
                                 virtual public ByteArrayDecoder {
 public:
  using BASE = DictDecoderImpl<ByteArrayType>;
  using BASE::DictDecoderImpl;

  int DecodeArrow(int num_values, int null_count, const uint8_t* valid_bits,
                  int64_t valid_bits_offset,
                  arrow::BinaryDictionary32Builder* builder) override {
    int result = 0;
    PARQUET_THROW_NOT_OK(DecodeArrow(num_values, null_count, valid_bits,
                                     valid_bits_offset, builder, &result));
    return result;
  }

  int DecodeArrowNonNull(int num_values,
                         arrow::BinaryDictionary32Builder* builder) override {
    int result = 0;
    PARQUET_THROW_NOT_OK(DecodeArrowNonNull(num_values, builder, &result));
    return result;
  }

  int DecodeArrow(int num_values, int null_count, const uint8_t* valid_bits,
                  int64_t valid_bits_offset, ArrowBinaryAccumulator* out) override {
    int result = 0;
    PARQUET_THROW_NOT_OK(DecodeArrowDense(num_values, null_count, valid_bits,
                                          valid_bits_offset, out, &result));
    return result;
  }

  int DecodeArrowNonNull(int num_values, ArrowBinaryAccumulator* out) override {
    int result = 0;
    PARQUET_THROW_NOT_OK(DecodeArrowDenseNonNull(num_values, out, &result));
    return result;
  }

 private:
  Status DecodeArrowDense(int num_values, int null_count, const uint8_t* valid_bits,
                          int64_t valid_bits_offset, ArrowBinaryAccumulator* out,
                          int* out_num_values) {
    constexpr int32_t buffer_size = 1024;
    int32_t indices_buffer[buffer_size];

    ArrowBinaryHelper helper(out);

    arrow::internal::BitmapReader bit_reader(valid_bits, valid_bits_offset, num_values);

    auto dict_values = reinterpret_cast<const ByteArray*>(dictionary_->data());
    int values_decoded = 0;
    int num_appended = 0;
    while (num_appended < num_values) {
      bool is_valid = bit_reader.IsSet();
      bit_reader.Next();

      if (is_valid) {
        int32_t batch_size =
            std::min<int32_t>(buffer_size, num_values - num_appended - null_count);
        int num_indices = idx_decoder_.GetBatch(indices_buffer, batch_size);

        int i = 0;
        while (true) {
          // Consume all indices
          if (is_valid) {
            const auto& val = dict_values[indices_buffer[i]];
            if (ARROW_PREDICT_FALSE(!helper.CanFit(val.len))) {
              RETURN_NOT_OK(helper.PushChunk());
            }
            RETURN_NOT_OK(helper.Append(val.ptr, static_cast<int32_t>(val.len)));
            ++i;
            ++values_decoded;
          } else {
            RETURN_NOT_OK(helper.AppendNull());
            --null_count;
          }
          ++num_appended;
          if (i == num_indices) {
            // Do not advance the bit_reader if we have fulfilled the decode
            // request
            break;
          }
          is_valid = bit_reader.IsSet();
          bit_reader.Next();
        }
      } else {
        RETURN_NOT_OK(helper.AppendNull());
        --null_count;
        ++num_appended;
      }
    }
    *out_num_values = values_decoded;
    return Status::OK();
  }

  Status DecodeArrowDenseNonNull(int num_values, ArrowBinaryAccumulator* out,
                                 int* out_num_values) {
    constexpr int32_t buffer_size = 2048;
    int32_t indices_buffer[buffer_size];
    int values_decoded = 0;

    ArrowBinaryHelper helper(out);
    auto dict_values = reinterpret_cast<const ByteArray*>(dictionary_->data());

    while (values_decoded < num_values) {
      int32_t batch_size = std::min<int32_t>(buffer_size, num_values - values_decoded);
      int num_indices = idx_decoder_.GetBatch(indices_buffer, batch_size);
      if (num_indices == 0) ParquetException::EofException();
      for (int i = 0; i < num_indices; ++i) {
        const auto& val = dict_values[indices_buffer[i]];
        if (ARROW_PREDICT_FALSE(!helper.CanFit(val.len))) {
          RETURN_NOT_OK(helper.PushChunk());
        }
        RETURN_NOT_OK(helper.Append(val.ptr, static_cast<int32_t>(val.len)));
      }
      values_decoded += num_indices;
    }
    *out_num_values = values_decoded;
    return Status::OK();
  }

  template <typename BuilderType>
  Status DecodeArrow(int num_values, int null_count, const uint8_t* valid_bits,
                     int64_t valid_bits_offset, BuilderType* builder,
                     int* out_num_values) {
    constexpr int32_t buffer_size = 1024;
    int32_t indices_buffer[buffer_size];

    RETURN_NOT_OK(builder->Reserve(num_values));
    arrow::internal::BitmapReader bit_reader(valid_bits, valid_bits_offset, num_values);

    auto dict_values = reinterpret_cast<const ByteArray*>(dictionary_->data());

    int values_decoded = 0;
    int num_appended = 0;
    while (num_appended < num_values) {
      bool is_valid = bit_reader.IsSet();
      bit_reader.Next();

      if (is_valid) {
        int32_t batch_size =
            std::min<int32_t>(buffer_size, num_values - num_appended - null_count);
        int num_indices = idx_decoder_.GetBatch(indices_buffer, batch_size);

        int i = 0;
        while (true) {
          // Consume all indices
          if (is_valid) {
            const auto& val = dict_values[indices_buffer[i]];
            RETURN_NOT_OK(builder->Append(val.ptr, val.len));
            ++i;
            ++values_decoded;
          } else {
            RETURN_NOT_OK(builder->AppendNull());
            --null_count;
          }
          ++num_appended;
          if (i == num_indices) {
            // Do not advance the bit_reader if we have fulfilled the decode
            // request
            break;
          }
          is_valid = bit_reader.IsSet();
          bit_reader.Next();
        }
      } else {
        RETURN_NOT_OK(builder->AppendNull());
        --null_count;
        ++num_appended;
      }
    }
    *out_num_values = values_decoded;
    return Status::OK();
  }

  template <typename BuilderType>
  Status DecodeArrowNonNull(int num_values, BuilderType* builder, int* out_num_values) {
    constexpr int32_t buffer_size = 2048;
    int32_t indices_buffer[buffer_size];
    int values_decoded = 0;
    RETURN_NOT_OK(builder->Reserve(num_values));

    auto dict_values = reinterpret_cast<const ByteArray*>(dictionary_->data());

    while (values_decoded < num_values) {
      int32_t batch_size = std::min<int32_t>(buffer_size, num_values - values_decoded);
      int num_indices = idx_decoder_.GetBatch(indices_buffer, batch_size);
      if (num_indices == 0) ParquetException::EofException();
      for (int i = 0; i < num_indices; ++i) {
        const auto& val = dict_values[indices_buffer[i]];
        RETURN_NOT_OK(builder->Append(val.ptr, val.len));
      }
      values_decoded += num_indices;
    }
    *out_num_values = values_decoded;
    return Status::OK();
  }
};

class DictFLBADecoder : public DictDecoderImpl<FLBAType>, virtual public FLBADecoder {
 public:
  using BASE = DictDecoderImpl<FLBAType>;
  using BASE::DictDecoderImpl;
};

// ----------------------------------------------------------------------
// DeltaBitPackDecoder

template <typename DType>
class DeltaBitPackDecoder : public DecoderImpl, virtual public TypedDecoder<DType> {
 public:
  typedef typename DType::c_type T;

  explicit DeltaBitPackDecoder(const ColumnDescriptor* descr,
                               MemoryPool* pool = arrow::default_memory_pool())
      : DecoderImpl(descr, Encoding::DELTA_BINARY_PACKED), pool_(pool) {
    if (DType::type_num != Type::INT32 && DType::type_num != Type::INT64) {
      throw ParquetException("Delta bit pack encoding should only be for integer data.");
    }
  }

  virtual void SetData(int num_values, const uint8_t* data, int len) {
    this->num_values_ = num_values;
    decoder_ = arrow::BitUtil::BitReader(data, len);
    values_current_block_ = 0;
    values_current_mini_block_ = 0;
  }

  virtual int Decode(T* buffer, int max_values) {
    return GetInternal(buffer, max_values);
  }

 private:
  void InitBlock() {
    int32_t block_size;
    if (!decoder_.GetVlqInt(&block_size)) ParquetException::EofException();
    if (!decoder_.GetVlqInt(&num_mini_blocks_)) ParquetException::EofException();
    if (!decoder_.GetVlqInt(&values_current_block_)) {
      ParquetException::EofException();
    }
    if (!decoder_.GetZigZagVlqInt(&last_value_)) ParquetException::EofException();

    delta_bit_widths_ = AllocateBuffer(pool_, num_mini_blocks_);
    uint8_t* bit_width_data = delta_bit_widths_->mutable_data();

    if (!decoder_.GetZigZagVlqInt(&min_delta_)) ParquetException::EofException();
    for (int i = 0; i < num_mini_blocks_; ++i) {
      if (!decoder_.GetAligned<uint8_t>(1, bit_width_data + i)) {
        ParquetException::EofException();
      }
    }
    values_per_mini_block_ = block_size / num_mini_blocks_;
    mini_block_idx_ = 0;
    delta_bit_width_ = bit_width_data[0];
    values_current_mini_block_ = values_per_mini_block_;
  }

  template <typename T>
  int GetInternal(T* buffer, int max_values) {
    max_values = std::min(max_values, this->num_values_);
    const uint8_t* bit_width_data = delta_bit_widths_->data();
    for (int i = 0; i < max_values; ++i) {
      if (ARROW_PREDICT_FALSE(values_current_mini_block_ == 0)) {
        ++mini_block_idx_;
        if (mini_block_idx_ < static_cast<size_t>(delta_bit_widths_->size())) {
          delta_bit_width_ = bit_width_data[mini_block_idx_];
          values_current_mini_block_ = values_per_mini_block_;
        } else {
          InitBlock();
          buffer[i] = last_value_;
          continue;
        }
      }

      // TODO: the key to this algorithm is to decode the entire miniblock at once.
      int64_t delta;
      if (!decoder_.GetValue(delta_bit_width_, &delta)) ParquetException::EofException();
      delta += min_delta_;
      last_value_ += static_cast<int32_t>(delta);
      buffer[i] = last_value_;
      --values_current_mini_block_;
    }
    this->num_values_ -= max_values;
    return max_values;
  }

  MemoryPool* pool_;
  arrow::BitUtil::BitReader decoder_;
  int32_t values_current_block_;
  int32_t num_mini_blocks_;
  uint64_t values_per_mini_block_;
  uint64_t values_current_mini_block_;

  int32_t min_delta_;
  size_t mini_block_idx_;
  std::shared_ptr<ResizableBuffer> delta_bit_widths_;
  int delta_bit_width_;

  int32_t last_value_;
};

// ----------------------------------------------------------------------
// DELTA_LENGTH_BYTE_ARRAY

class DeltaLengthByteArrayDecoder : public DecoderImpl,
                                    virtual public TypedDecoder<ByteArrayType> {
 public:
  explicit DeltaLengthByteArrayDecoder(const ColumnDescriptor* descr,
                                       MemoryPool* pool = arrow::default_memory_pool())
      : DecoderImpl(descr, Encoding::DELTA_LENGTH_BYTE_ARRAY),
        len_decoder_(nullptr, pool) {}

  virtual void SetData(int num_values, const uint8_t* data, int len) {
    num_values_ = num_values;
    if (len == 0) return;
    int total_lengths_len = arrow::util::SafeLoadAs<int32_t>(data);
    data += 4;
    this->len_decoder_.SetData(num_values, data, total_lengths_len);
    data_ = data + total_lengths_len;
    this->len_ = len - 4 - total_lengths_len;
  }

  virtual int Decode(ByteArray* buffer, int max_values) {
    max_values = std::min(max_values, num_values_);
    std::vector<int> lengths(max_values);
    len_decoder_.Decode(lengths.data(), max_values);
    for (int i = 0; i < max_values; ++i) {
      buffer[i].len = lengths[i];
      buffer[i].ptr = data_;
      this->data_ += lengths[i];
      this->len_ -= lengths[i];
    }
    this->num_values_ -= max_values;
    return max_values;
  }

 private:
  DeltaBitPackDecoder<Int32Type> len_decoder_;
};

// ----------------------------------------------------------------------
// DELTA_BYTE_ARRAY

class DeltaByteArrayDecoder : public DecoderImpl,
                              virtual public TypedDecoder<ByteArrayType> {
 public:
  explicit DeltaByteArrayDecoder(const ColumnDescriptor* descr,
                                 MemoryPool* pool = arrow::default_memory_pool())
      : DecoderImpl(descr, Encoding::DELTA_BYTE_ARRAY),
        prefix_len_decoder_(nullptr, pool),
        suffix_decoder_(nullptr, pool),
        last_value_(0, nullptr) {}

  virtual void SetData(int num_values, const uint8_t* data, int len) {
    num_values_ = num_values;
    if (len == 0) return;
    int prefix_len_length = arrow::util::SafeLoadAs<int32_t>(data);
    data += 4;
    len -= 4;
    prefix_len_decoder_.SetData(num_values, data, prefix_len_length);
    data += prefix_len_length;
    len -= prefix_len_length;
    suffix_decoder_.SetData(num_values, data, len);
  }

  // TODO: this doesn't work and requires memory management. We need to allocate
  // new strings to store the results.
  virtual int Decode(ByteArray* buffer, int max_values) {
    max_values = std::min(max_values, this->num_values_);
    for (int i = 0; i < max_values; ++i) {
      int prefix_len = 0;
      prefix_len_decoder_.Decode(&prefix_len, 1);
      ByteArray suffix = {0, nullptr};
      suffix_decoder_.Decode(&suffix, 1);
      buffer[i].len = prefix_len + suffix.len;

      uint8_t* result = reinterpret_cast<uint8_t*>(malloc(buffer[i].len));
      memcpy(result, last_value_.ptr, prefix_len);
      memcpy(result + prefix_len, suffix.ptr, suffix.len);

      buffer[i].ptr = result;
      last_value_ = buffer[i];
    }
    this->num_values_ -= max_values;
    return max_values;
  }

 private:
  DeltaBitPackDecoder<Int32Type> prefix_len_decoder_;
  DeltaLengthByteArrayDecoder suffix_decoder_;
  ByteArray last_value_;
};


// ----------------------------------------------------------------------
// BYTE_STREAM_SPLIT

template <typename DType>
class FPByteStreamSplitDecoder : public DecoderImpl, virtual public TypedDecoder<DType> {
 public:
  using T = typename DType::c_type;
  explicit FPByteStreamSplitDecoder(const ColumnDescriptor* descr);

  int Decode(T* buffer, int max_values) override;

  void SetData(int num_values, const uint8_t* data, int len) override;

 private:
  int streamSizeInBytes_ { 0U };
  int numValuesDecoded_ { 0U };
  std::shared_ptr<ResizableBuffer> tmp_buffer;
};

template<typename DType>
FPByteStreamSplitDecoder<DType>::FPByteStreamSplitDecoder(const ColumnDescriptor* descr)
    : DecoderImpl(descr, Encoding::BYTE_STREAM_SPLIT) {
}

template<typename DType>
void FPByteStreamSplitDecoder<DType>::SetData(int num_values, const uint8_t* data, int len) {
  PARQUET_THROW_NOT_OK(arrow::AllocateResizableBuffer(arrow::default_memory_pool(),
                                                        num_values * sizeof(T), &tmp_buffer));
  arrow::util::RleDecoder rleDecoder = arrow::util::RleDecoder(data, len, 8);
  const int numDecoded = rleDecoder.GetBatch(tmp_buffer->mutable_data(), num_values * sizeof(T));
  DecoderImpl::SetData(num_values, tmp_buffer->mutable_data(), numDecoded);
  streamSizeInBytes_ = num_values;
  numValuesDecoded_ = 0;
}

template<typename DType>
int FPByteStreamSplitDecoder<DType>::Decode(T* buffer, int max_values) {
  constexpr size_t numStreams = sizeof(T);
  using UnsignedType = typename UnsignedTypeWithWidth<sizeof(T)>::type;
  const int valuesToDecode = std::min(num_values_ - numValuesDecoded_, max_values);
  const uint8_t* tmp_buffer_data = tmp_buffer->mutable_data();
  for (int i = 0; i < valuesToDecode; ++i) {
    UnsignedType reconstructedValueAsUint { 0U };
    for (size_t b = 0; b < numStreams; ++b) {
      const size_t byteIndex = b * streamSizeInBytes_ + numValuesDecoded_ + i;
      const UnsignedType byteValue = static_cast<UnsignedType>(tmp_buffer_data[byteIndex]);
      reconstructedValueAsUint |= (byteValue << (8U * b));
    }
    const T reconstructedValueAsFP = *reinterpret_cast<const T*>(&reconstructedValueAsUint);
    buffer[i] = reconstructedValueAsFP;
  }
  numValuesDecoded_ += valuesToDecode;
  return valuesToDecode;
}

// ----------------------------------------------------------------------

std::unique_ptr<Decoder> MakeDecoder(Type::type type_num, Encoding::type encoding,
                                     const ColumnDescriptor* descr) {
  if (encoding == Encoding::PLAIN) {
    switch (type_num) {
      case Type::BOOLEAN:
        return std::unique_ptr<Decoder>(new PlainBooleanDecoder(descr));
      case Type::INT32:
        return std::unique_ptr<Decoder>(new PlainDecoder<Int32Type>(descr));
      case Type::INT64:
        return std::unique_ptr<Decoder>(new PlainDecoder<Int64Type>(descr));
      case Type::INT96:
        return std::unique_ptr<Decoder>(new PlainDecoder<Int96Type>(descr));
      case Type::FLOAT:
        return std::unique_ptr<Decoder>(new PlainDecoder<FloatType>(descr));
      case Type::DOUBLE:
        return std::unique_ptr<Decoder>(new PlainDecoder<DoubleType>(descr));
      case Type::BYTE_ARRAY:
        return std::unique_ptr<Decoder>(new PlainByteArrayDecoder(descr));
      case Type::FIXED_LEN_BYTE_ARRAY:
        return std::unique_ptr<Decoder>(new PlainFLBADecoder(descr));
      default:
        break;
    }
  } else if (encoding == Encoding::BYTE_STREAM_SPLIT) {
    switch (type_num) {
      case Type::FLOAT:
        return std::unique_ptr<Decoder>(new FPByteStreamSplitDecoder<FloatType>(descr));
      case Type::DOUBLE:
        return std::unique_ptr<Decoder>(new FPByteStreamSplitDecoder<DoubleType>(descr));
      default:
        break;
    }
  } else {
    ParquetException::NYI("Selected encoding is not supported");
  }
  DCHECK(false) << "Should not be able to reach this code";
  return nullptr;
}

namespace detail {

std::unique_ptr<Decoder> MakeDictDecoder(Type::type type_num,
                                         const ColumnDescriptor* descr,
                                         MemoryPool* pool) {
  switch (type_num) {
    case Type::BOOLEAN:
      ParquetException::NYI("Dictionary encoding not implemented for boolean type");
    case Type::INT32:
      return std::unique_ptr<Decoder>(new DictDecoderImpl<Int32Type>(descr, pool));
    case Type::INT64:
      return std::unique_ptr<Decoder>(new DictDecoderImpl<Int64Type>(descr, pool));
    case Type::INT96:
      return std::unique_ptr<Decoder>(new DictDecoderImpl<Int96Type>(descr, pool));
    case Type::FLOAT:
      return std::unique_ptr<Decoder>(new DictDecoderImpl<FloatType>(descr, pool));
    case Type::DOUBLE:
      return std::unique_ptr<Decoder>(new DictDecoderImpl<DoubleType>(descr, pool));
    case Type::BYTE_ARRAY:
      return std::unique_ptr<Decoder>(new DictByteArrayDecoderImpl(descr, pool));
    case Type::FIXED_LEN_BYTE_ARRAY:
      return std::unique_ptr<Decoder>(new DictFLBADecoder(descr, pool));
    default:
      break;
  }
  DCHECK(false) << "Should not be able to reach this code";
  return nullptr;
}

}  // namespace detail
}  // namespace parquet
