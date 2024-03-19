// https://docs.opencv.org/4.x/d3/d63/classcv_1_1Mat.html
export enum BitDepth {
  UINT8 = 0,
  INT8 = 1,
  UINT16 = 2,
  INT16 = 3,
  INT32 = 4,
  FLOAT32 = 5,
  FLOAT64 = 6,
  FLOAT16 = 7,
}

export interface ImageDimensions {
  width: number
  height: number
  stride: number
  channels: number
  bit_depth: BitDepth
}

export const unmarshal_image_dim_info = (data: ArrayBufferLike): ImageDimensions => {
  const view = new DataView(data)
  const width = view.getUint16(0, false)
  const height = view.getUint16(2, false)
  const stride = view.getUint16(4, false)
  const channels = view.getUint8(6)
  const bit_depth = view.getUint8(7)
  return { width, height, stride, channels, bit_depth }
}

export const SIZE_NEEDED = 8
