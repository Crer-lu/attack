import { createApp } from 'vue'
import axios from '@/plugins/axiosInstance.js'
import antd from 'ant-design-vue'

import App from './App.vue'

const app=createApp(App)
app.use(antd)
app.mount('#app')
app.config.globalProperties.$axios=axios
