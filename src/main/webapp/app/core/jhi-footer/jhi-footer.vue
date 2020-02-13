<template>
    <div id="footer" class="footer">
        <p v-text="$t('footer')"></p>
    </div>
</template>
<script lang="ts" src="./jhi-footer.component.ts">
export default {
  props: {
    backgroundColor: String,
    type: String
  },
  data() {
    return {
      year: new Date().getFullYear()
    };
  }
};
</script>

<style scoped>
    .footer {
        text-align: left;
        background-color: white;
        height: fit-content;
    }
</style>
