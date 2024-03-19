<script lang="ts">
import { onMount } from "svelte"
import { unmarshal_image_dim_info, SIZE_NEEDED } from "./models/image"
import type { ImageDimensions } from "./models/image"

let video_dim: ImageDimensions | undefined

const ws_addr = "ws://localhost:8000"

let videoElem: HTMLVideoElement

onMount(() => {
  const conn = new WebSocket(ws_addr)
  conn.binaryType = "arraybuffer"
  conn.onopen = () => {
    console.info("WebSocket connection established")
  }
  const canvas = document.createElement("canvas")
  const ctx = canvas.getContext("2d")
  // might be set a fixed frame rate or leave it empty to calculate it dynamically
  // https://developer.mozilla.org/en-US/docs/Web/API/HTMLCanvasElement/captureStream
  // https://developer.chrome.com/blog/capture-stream
  const stream = canvas.captureStream()
  videoElem.srcObject = stream
  conn.onmessage = (e) => {
    const data = e.data
    if (data instanceof ArrayBuffer) {
      const dim_info = unmarshal_image_dim_info(data.slice(0, SIZE_NEEDED))
      if (
        canvas.width !== dim_info.width ||
        canvas.height !== dim_info.height
      ) {
        canvas.width = dim_info.width
        canvas.height = dim_info.height
      }
      video_dim = dim_info
      const img_data = new Uint8ClampedArray(data.slice(SIZE_NEEDED))
      // https://developer.mozilla.org/en-US/docs/Web/API/ImageData
      const img = ctx?.createImageData(dim_info.width, dim_info.height)
      if (img) {
        for (let i = 0, j = 0; i < img_data.length; i += 3, j += 4) {
          img.data[j] = img_data[i]
          img.data[j + 1] = img_data[i + 1]
          img.data[j + 2] = img_data[i + 2]
          // image data is always 4 bytes per pixel (the last byte is alpha channel, set to 255 for opaque image)
          img.data[j + 3] = 255
        }
        // https://developer.mozilla.org/en-US/docs/Web/API/CanvasRenderingContext2D/putImageData
        ctx?.putImageData(img, 0, 0)
      }
    }
  }
  return () => {
    conn.close()
  }
})
</script>

<main>
  <div class="bg-lime-400">test</div>
  {#if video_dim}
    <p>
      {video_dim.width}x{video_dim.height}
    </p>
  {/if}
  <video
    id="the-video"
    autoplay
    controls
    bind:this={videoElem}
    width={video_dim ? video_dim.width : 0}
    height={video_dim ? video_dim.height : 0}
  >
    Your browser does not support the video tag.
  </video>
</main>
