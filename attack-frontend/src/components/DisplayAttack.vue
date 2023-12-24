<template>
  <a-layout class="layout">
    <a-layout-content>
      <a-row
        type="flex"
        justify="center"
        align="middle"
        class="full-height"
        style="position: relative"
      >
        <a-col>
          <a-upload
            list-type="picture-card"
            :before-upload="beforeUpload"
            class="upload-area"
            @change="handleChange"
          >
            <div v-if="length == 0">
              <div>上传图片</div>
            </div>
          </a-upload>
        </a-col>
        <a-col><div style="width: 20px"></div></a-col>
        <a-col>
          <div v-if="convertedImageUrl.length > 0" class="result-area" style="">
            <div
              style="
                border: 1px solid #d9d9d9;
                border-radius: 6px;
                height: 100%;
                width: 100%;
                padding: 8px;
                position: relative;
              "
            >
              <img
                :src="convertedImageUrl"
                style="
                  position: relative;
                  width: 100%;
                  height: 100%;
                  object-fit: cover;
                "
                alt="转换后的图片"
                class="converted-image"
              />
            </div>
          </div>
        </a-col>
        <a-col><div style="width: 20px"></div></a-col>
        <a-col class="text-center">
          <div
            style="
              padding: 10px;
              border: 1px dashed #d9d9d9;
              border-radius: 4px;
            "
          >
            <div>User Tab</div>
            <a-button @click="convertImage" class="action-button"
              >转换</a-button
            >
            <a-button @click="recognizeImage" class="action-button"
              >识别</a-button
            >
            <div v-if="recognizedText.length > 0">
              <p>识别结果：{{ recognizedText }}</p>
            </div>
          </div>
          <br />
          <a-col>
            <div
              style="
                padding: 10px;
                border: 1px dashed #d9d9d9;
                border-radius: 4px;
              "
            >
              <div>Hacker Tab</div>
              <a-select
                style="width: 240px; margin: 10px"
                @change="handleSelect"
              >
                <a-select-option value="black_latent_class"
                  >黑盒 类别 语义向量干扰</a-select-option
                >
                <a-select-option value="black_latent_mse"
                  >黑盒 似然 语义向量干扰</a-select-option
                >
                <a-select-option value="black_image_class"
                  >黑盒 类别 图像数据干扰</a-select-option
                >
                <a-select-option value="black_image_class_condition"
                  >黑盒 类别 定向 图像数据干扰</a-select-option
                >
                <a-select-option value="while_latent_class"
                  >白盒 类别 语义向量干扰</a-select-option
                >
                <a-select-option value="while_image_class"
                  >白盒 类别 图像数据干扰</a-select-option
                >
                <a-select-option value="while_image_class_conition"
                  >白盒 类别 定向 图像数据干扰</a-select-option
                >
                <a-select-option value="while_image_class_chinese"
                  >白盒 类别 定向 汉字图像数据干扰</a-select-option
                >
                <a-select-option value="none">无</a-select-option>
              </a-select>
              <div v-if="directed" style="width: 10px"></div>
              <a-select
                v-if="directed"
                style="width: 240px; margin: 10px"
                @change="handleCondition"
              >
                <a-select-option value="0">0</a-select-option>
                <a-select-option value="1">1</a-select-option>
                <a-select-option value="2">2</a-select-option>
                <a-select-option value="3">3</a-select-option>
                <a-select-option value="4">4</a-select-option>
                <a-select-option value="5">5</a-select-option>
                <a-select-option value="6">6</a-select-option>
                <a-select-option value="7">7</a-select-option>
                <a-select-option value="8">8</a-select-option>
                <a-select-option value="9">9</a-select-option>
              </a-select>
            </div>
          </a-col>
        </a-col>
      </a-row>
    </a-layout-content>
  </a-layout>
</template>

<script>
import API from "../plugins/axiosInstance";
import { message } from "ant-design-vue";
export default {
  name: "ImageConverter",
  data() {
    return {
      convertedImageUrl: "",
      recognizedText: "",
      formData: null,
      noiseMode: "none",
      length: 0,
      directed: false,
      condition: 0,
    };
  },
  methods: {
    handleCondition(info) {
      this.condition = Number(info);
    },
    handleSelect(info) {
      this.noiseMode = info;
      if (
        info == "while_image_class_conition" ||
        info == "black_image_class_condition"
      ) {
        this.directed = true;
      } else {
        this.directed = false;
      }
    },
    beforeUpload(file) {
      // 处理上传文件
      if (file.type !== "image/jpeg" && file.type !== "image/png")
        message.error("只能上传 JPG/PNG 文件！");
      return false;
    },
    handleChange(info) {
      this.formData = new FormData();
      if (info.fileList.length > 0) {
        const file = info.file;
        const status = file.status;
        this.formData.append("file", info.fileList[0].originFileObj);
        if (status !== "uploading") {
          console.log(info.file, info.fileList);
        }
        if (status === "done") {
          message.success(`${info.file.name} file uploaded successfully.`);
        } else if (status === "error") {
          message.error(`${info.file.name} file upload failed.`);
        }
        this.length = 1;
      } else {
        this.length = 0;
      }
    },
    convertImage() {
      // 调用后端接口进行图片转换
      if (!this.formData) {
        // 如果没有文件，显示错误或退出
        return;
      }
      if (this.noiseMode == "while_image_class_chinese") {
        API({
          method: "post",
          url: "/api/convert_noisy",
          data: this.formData,
        })
          .then((response) => {
            var local_url = response.data.url;
            local_url = local_url.substr(local_url.lastIndexOf("/") + 1);
            this.convertedImageUrl = 'https://2n647o0588.imdo.co' + "/media/" + local_url;
          })
          .catch((error) => {
            console.log(error);
          });
      } else {
        API({
          method: "post",
          url: "/api/convert",
          data: this.formData,
        })
          .then((response) => {
            var local_url = response.data.url;
            local_url = local_url.substr(local_url.lastIndexOf("/") + 1);
            this.convertedImageUrl = 'https://2n647o0588.imdo.co' + "/media/" + local_url;
          })
          .catch((error) => {
            console.log(error);
          });
      }
    },
    recognizeImage() {
      // 调用后端接口进行图片识别
      if (!this.formData) {
        // 如果没有文件，显示错误或退出
        return;
      }
      API({
        method: "post",
        url: "/api/recognize",
        data: {
          path: this.convertedImageUrl,
          noise: this.noiseMode,
          condition: this.condition,
        },
      })
        .then((response) => {
          this.recognizedText = response.data.text;
        })
        .catch((error) => {
          console.log(error);
        });
    },
  },
};
</script>

<style scoped>
.layout {
  height: 100vh;
}

.full-height {
  height: 100%;
}

.result-area {
  position: relative;
  width: 160px;
  /* 方框宽度 */
  height: 160px;
  /* 方框高度，与宽度相同 */
  border: 1px dashed #d9d9d9;
  border-radius: 4px;
  display: flex;
  justify-content: center;
  align-items: center;
  margin: auto;
  padding: 30px;
}

.upload-area {
  width: 160px;
  /* 方框宽度 */
  height: 160px;
  /* 方框高度，与宽度相同 */
  border: 1px dashed #d9d9d9;
  border-radius: 4px;
  display: flex;
  justify-content: center;
  align-items: center;
  margin: auto;

  /* 水平居中 */
}

.text-center {
  text-align: center;
}

.action-button {
  margin: 10px;
  width: 100px;
}

.converted-image {
  max-width: 100%;
  height: auto;
}
</style>
