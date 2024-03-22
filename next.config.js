/** @type {import('next').NextConfig} */
const nextConfig = { 
  experimental: {
    typedRoutes: true,
  },
  webpack(config) {
    config.externals = config.externals || [];
    config.externals = [...config.externals, "hnswlib-node"]
    config.resolve.alias['fs'] = false;

    return config
  },

}

module.exports = nextConfig
