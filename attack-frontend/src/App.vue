<template>
  <ImageConverter />
</template>

<script>
import ImageConverter from './components/DisplayAttack.vue'

export default {
  name: 'App',
  components: {
    ImageConverter
  }
}
</script>

<style>
#app {
  font-family: Avenir, Helvetica, Arial, sans-serif;
  -webkit-font-smoothing: antialiased;
  -moz-osx-font-smoothing: grayscale;
  text-align: center;
  color: #2c3e50;
  margin: 0 auto;
  height: 100%;
}
</style>
