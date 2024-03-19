<script lang="ts">
import { onMount } from "svelte"
import type {
  Output as WorkerOutput,
  Input as WorkerInput,
} from "./worker/process"
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
  const worker = new Worker(new URL("./worker/process.ts", import.meta.url))
  const to_workder = (data: WorkerInput) => {
    worker.postMessage(data, [data])
  }
  videoElem.srcObject = stream
  conn.onmessage = (e) => {
    const data = e.data
    if (data instanceof ArrayBuffer) {
      to_workder(data)
    }
  }
  worker.onmessage = (e: MessageEvent<WorkerOutput>) => {
    const [img, dims] = e.data
    if (canvas.width !== dims.width || canvas.height !== dims.height) {
      canvas.width = dims.width
      canvas.height = dims.height
    }
    ctx?.putImageData(img, 0, 0)
  }
  return () => {
    conn.close()
    worker.terminate()
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
