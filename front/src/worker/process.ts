import type { ImageDimensions } from "../models/image"
import { unmarshal_image_dim_info, SIZE_NEEDED } from "../models/image"

// the raw data wrapped with header
export type Input = ArrayBufferLike
export type Output = [ImageData, ImageDimensions]

self.onmessage = (event: MessageEvent<Input>) => {
  const data = event.data
  const dim_info = unmarshal_image_dim_info(data.slice(0, SIZE_NEEDED))
  const img_data = new Uint8ClampedArray(data.slice(SIZE_NEEDED))
  const img = new ImageData(dim_info.width, dim_info.height)
  // https://developer.mozilla.org/en-US/docs/Web/API/CanvasRenderingContext2D/createImageData
  // https://developer.mozilla.org/en-US/docs/Web/API/ImageData/data
  // https://developer.mozilla.org/en-US/docs/Web/API/ImageData
  for (let i = 0, j = 0; i < img_data.length; i += 3, j += 4) {
    img.data[j] = img_data[i]
    img.data[j + 1] = img_data[i + 1]
    img.data[j + 2] = img_data[i + 2]
    // image data is always 4 bytes per pixel (the last byte is alpha channel, set to 255 for opaque image)
    img.data[j + 3] = 255
  }
  // @ts-ignore - the buffer is transferred to the main thread
  self.postMessage([img, dim_info] as Output, [img.data.buffer])
}
