/** @type {import('next').NextConfig} */
const nextConfig = {
  reactStrictMode: true,
  // onnxruntime-web and MediaPipe load their own WASM; keep them out of the server bundle.
  serverExternalPackages: ["onnxruntime-web"],
};
export default nextConfig;
