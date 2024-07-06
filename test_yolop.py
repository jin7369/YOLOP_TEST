import torch
import cv2
import numpy as np
from YOLOP_TEST.lib.models.YOLOP import MCnet, YOLOP
import time
class Model_YOLOP:
    def __init__(self):
        self.model = MCnet(YOLOP)
        self.model.load_state_dict(torch.load('model'))


    def predict(self, img):
        # 입력 : 이미지 numpy 배열
        # 출력 : 이미지 numpy 배열


        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # BGR -> RGB
        img = cv2.resize(img, (640, 640))
        # 입력 크기를 맞춰줌

        # 이미지를 float 타입으로 변환하고 PyTorch 텐서로 변환
        img_tensor = torch.tensor(img, dtype=torch.float32)
        img_tensor = img_tensor.permute(2, 0, 1).unsqueeze(0)
        _, _, ll_seg_out = self.model(img_tensor)

        # da_seg_out을 CPU로 옮기고 numpy 배열로 변환하여 이미지로 표시
        ll_seg_out_np = ll_seg_out.cpu().detach().numpy()
        ll_seg_out_np = ll_seg_out_np[0, 1]  # 두 번째 채널 선택
        ll_seg_out_np = (ll_seg_out_np * 255).astype(np.uint8)
        return ll_seg_out_np



if __name__ == '__main__':
    yolop = Model_YOLOP()
    img = cv2.imread('img.png')
    # YOLOP 모델을 사용하여 예측 수행
    start = time.time()
    ll_seg_out = yolop.predict(img)
    end = time.time()
    print(end - start)
    cv2.imshow("Output", ll_seg_out)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


