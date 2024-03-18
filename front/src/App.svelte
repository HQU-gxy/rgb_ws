<script lang="ts">
  import { onMount } from "svelte";

  // https://docs.opencv.org/4.x/d3/d63/classcv_1_1Mat.html
  enum BitDepth {
    UINT8 = 0,
    INT8 = 1,
    UINT16 = 2,
    INT16 = 3,
    INT32 = 4,
    FLOAT32 = 5,
    FLOAT64 = 6,
    FLOAT16 = 7,
  }

  interface ImageDimensions {
    width: number;
    height: number;
    stride: number;
    channels: number;
    bit_depth: BitDepth;
  }

  let video_dim: ImageDimensions | undefined = undefined;

  const unmarshal_image_dim_info = (data: ArrayBufferLike): ImageDimensions => {
    const view = new DataView(data);
    const width = view.getUint16(0, false);
    const height = view.getUint16(2, false);
    const stride = view.getUint16(4, false);
    const channels = view.getUint8(6);
    const bit_depth = view.getUint8(7);
    return { width, height, stride, channels, bit_depth };
  };

  const image_dim_info_size_needed = 8;

  const ws_addr = "ws://localhost:8000";

  let videoElem: HTMLVideoElement;

  onMount(() => {
    const conn = new WebSocket(ws_addr);
    conn.binaryType = "arraybuffer";
    conn.onopen = () => {
      console.info("WebSocket connection established");
    };
    const canvas = document.createElement("canvas");
    const ctx = canvas.getContext("2d");
    // might be set a fixed frame rate or leave it empty to calculate it dynamically
    const stream = canvas.captureStream();
    videoElem.srcObject = stream;
    conn.onmessage = (e) => {
      const data = e.data;
      if (data instanceof ArrayBuffer) {
        const dim_info = unmarshal_image_dim_info(
          data.slice(0, image_dim_info_size_needed),
        );
        canvas.width = dim_info.width;
        canvas.height = dim_info.height;
        video_dim = dim_info;
        const img_data = new Uint8ClampedArray(
          data.slice(image_dim_info_size_needed),
        );
        const img = ctx?.createImageData(dim_info.width, dim_info.height);
        if (img) {
          for (let i = 0, j = 0; i < img_data.length; i += 3, j += 4) {
            img.data[j] = img_data[i];
            img.data[j + 1] = img_data[i + 1];
            img.data[j + 2] = img_data[i + 2];
            img.data[j + 3] = 255;
          }
          ctx?.putImageData(img, 0, 0);
        }
      }
    };
    return () => {
      conn.close();
    };
  });
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
  ></video>
</main>
