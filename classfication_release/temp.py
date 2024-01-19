from fvcore.nn import FlopCountAnalysis
from fvcore.nn import flop_count_table
import torch
import torch.nn as nn
import numpy as np
import time
from einops import einsum, rearrange
import torch.nn.functional as F
from AutoOpInspect import OpsInfoProvider
from typing import Optional, Tuple, Union


class attention(nn.Module):
    def __init__(self, embed_dim, num_heads, h, w):
        super().__init__()
        self.num_heads = num_heads
        self.embed_dim = embed_dim
        self.h, self.w = h, w
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=True)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=True)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=True)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=True)

    def forward(self, x):
        x = x.flatten(1, 2)
        q = self.q_proj(x) #bsz, HW, dim
        k = self.k_proj(x) #bsz, HW, dim
        v = self.v_proj(x) #bsz, HW, dim

        q = q.view(1, self.h*self.w, self.num_heads, self.embed_dim//self.num_heads).permute(0, 2, 1, 3) #(b n hw d1)
        k = k.view(1, self.h*self.w, self.num_heads, self.embed_dim//self.num_heads).permute(0, 2, 1, 3) #(b n hw d1)
        v = v.view(1, self.h*self.w, self.num_heads, self.embed_dim//self.num_heads).permute(0, 2, 1, 3) #(b n hw d1)

        attn_score = q @ k.transpose(-1, -2) #(b h n hw hw)
        attn_score = torch.softmax(attn_score, -1) #(b n hw hw)
        output = torch.matmul(attn_score, v) #(b h n w d2)

        output = output.permute(0, 2, 1, 3).flatten(2,3) #(b h w n*d2)
        output = self.out_proj(output)
        return output

        
class linear_attention(nn.Module):
    def __init__(self, embed_dim, num_heads, h, w):
        super().__init__()
        self.num_heads = num_heads
        self.embed_dim = embed_dim
        self.h, self.w = h, w
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=True)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=True)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=True)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=True)
        self.act = nn.ReLU()

    def forward(self, x):
        x = x.flatten(1, 2)
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        q = q.view(1, self.h*self.w, self.num_heads, self.embed_dim//self.num_heads).permute(0, 2, 1, 3) #(b n hw d1)
        k = k.view(1, self.h*self.w, self.num_heads, self.embed_dim//self.num_heads).permute(0, 2, 1, 3) #(b n hw d1)
        v = v.view(1, self.h*self.w, self.num_heads, self.embed_dim//self.num_heads).permute(0, 2, 1, 3) #(b n hw d1)

        q = self.act(q)
        k = self.act(k)
        
        output = k.transpose(-1, -2) @ v
        output = q @ output

        output = output.permute(0, 2, 1, 3).flatten(2,3) #(b h w n*d2)
        output = self.out_proj(output)
        return output
    
class retention(nn.Module):
    def __init__(self, embed_dim, num_heads, h, w):
        super().__init__()
        self.num_heads = num_heads
        self.embed_dim = embed_dim
        self.h, self.w = h, w
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=True)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=True)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=True)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=True)

    def forward(self, x):
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        q = q.view(1, self.h, self.w, self.num_heads, self.embed_dim//self.num_heads).permute(0, 3, 1, 2, 4) #(b n h w d1)
        k = k.view(1, self.h, self.w, self.num_heads, self.embed_dim//self.num_heads).permute(0, 3, 1, 2, 4) #(b n h w d1)

        qr_w = q.transpose(1, 2) #(b h n w d1)
        kr_w = k.transpose(1, 2) #(b h n w d1)
        v = v.reshape(1, self.h, self.w, self.num_heads, -1).permute(0, 1, 3, 2, 4) #(b h n w d2)

        qk_mat_w = qr_w @ kr_w.transpose(-1, -2) #(b h n w w)
        qk_mat_w = torch.softmax(qk_mat_w, -1) #(b h n w w)
        v = torch.matmul(qk_mat_w, v) #(b h n w d2)


        qr_h = q.permute(0, 3, 1, 2, 4) #(b w n h d1)
        kr_h = k.permute(0, 3, 1, 2, 4) #(b w n h d1)
        v = v.permute(0, 3, 2, 1, 4) #(b w n h d2)

        qk_mat_h = qr_h @ kr_h.transpose(-1, -2) #(b w n h h)
        qk_mat_h = torch.softmax(qk_mat_h, -1) #(b w n h h)
        output = torch.matmul(qk_mat_h, v) #(b w n h d2)
        
        output = output.permute(0, 3, 1, 2, 4).flatten(-2, -1) #(b h w n*d2)
        output = self.out_proj(output)
        return output.flatten(1,2)

class linear_retention(nn.Module):
    def __init__(self, embed_dim, num_heads, h, w):
        super().__init__()
        self.num_heads = num_heads
        self.embed_dim = embed_dim
        self.h, self.w = h, w
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=True)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=True)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=True)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=True)
        self.act = nn.ReLU()

    def forward(self, x):
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        q = q.view(1, self.h, self.w, self.num_heads, self.embed_dim//self.num_heads).permute(0, 3, 1, 2, 4) #(b n h w d1)
        k = k.view(1, self.h, self.w, self.num_heads, self.embed_dim//self.num_heads).permute(0, 3, 1, 2, 4) #(b n h w d1)
        v = v.view(1, self.h, self.w, self.num_heads, self.embed_dim//self.num_heads).permute(0, 3, 1, 2, 4) #(b n h w d1)

        q = self.act(q)
        k = self.act(k)

        qr_w = q.transpose(1, 2) #(b h n w d1)
        kr_w = k.transpose(1, 2) #(b h n w d1)
        v = v.transpose(1, 2) #(b h n w d1)

        kv = kr_w.transpose(-1, -2) @ v #(b h n d1 d1)
        output = torch.matmul(qr_w, kv) #(b h n w d1)

        qr_h = q.permute(0, 3, 1, 2, 4) #(b w n h d1)
        kr_h = k.permute(0, 3, 1, 2, 4) #(b w n h d1)
        v = v.transpose(1, 3) #(b w n h d1)

        kv = kr_h.transpose(-1, -2) @ v #(b w n d1 d1)
        output = output.transpose(1, 3) #(b w n h d1)
        output = torch.matmul(output, kv) #(b w n h d)
        output = output.permute(0, 3, 1, 2, 4).flatten(-2, -1) #(b h w n*d2)
        output = self.out_proj(output)
        return output.flatten(1,2)


# class linear_retention(nn.Module):
#     def __init__(self, embed_dim, num_heads, h, w):
#         super().__init__()
#         self.num_heads = num_heads
#         self.embed_dim = embed_dim
#         self.h, self.w = h, w
#         self.q_proj = nn.Linear(embed_dim, embed_dim, bias=True)
#         self.k_proj = nn.Linear(embed_dim, embed_dim, bias=True)
#         self.v_proj = nn.Linear(embed_dim, embed_dim, bias=True)
#         self.out_proj = nn.Linear(embed_dim, embed_dim, bias=True)
#         self.act = nn.ReLU()

#     def forward(self, x):
#         q = self.q_proj(x)
#         k = self.k_proj(x)
#         v = self.v_proj(x)

#         q = q.view(1, self.h, self.w, self.num_heads, self.embed_dim//self.num_heads).permute(0, 3, 1, 2, 4) #(b n h w d1)
#         k = k.view(1, self.h, self.w, self.num_heads, self.embed_dim//self.num_heads).permute(0, 3, 1, 2, 4) #(b n h w d1)
#         v = v.view(1, self.h, self.w, self.num_heads, self.embed_dim//self.num_heads).permute(0, 3, 1, 2, 4) #(b n h w d1)

#         q = self.act(q)
#         k = self.act(k)

#         qr_w = q.transpose(1, 2) #(b h n w d1)
#         kr_w = k.transpose(1, 2) #(b h n w d1)
#         v = v.transpose(1, 2) #(b h n w d1)

#         kv = kr_w.transpose(-1, -2) @ v #(b h n d1 d1)
#         output = torch.matmul(qr_w, kv) #(b h n w d1)

#         qr_h = q.permute(0, 3, 1, 2, 4) #(b w n h d1)
#         kr_h = k.permute(0, 3, 1, 2, 4) #(b w n h d1)
#         v = v.transpose(1, 3) #(b w n h d1)

#         kv = kr_h.transpose(-1, -2) @ v #(b w n d1 d1)
#         output = output.transpose(1, 3) #(b w n h d1)
#         output = torch.matmul(output, kv) #(b w n h d)
#         output = output.permute(0, 3, 1, 2, 4).flatten(-2, -1) #(b h w n*d2)
#         output = self.out_proj(output)
#         return output.flatten(1,2)

def measure_inf_time(model, input_data):
    model.eval()
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    start_time = time.time()

    with torch.no_grad():
        output = model(input_data)
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    end_time = time.time()

    inference_time = end_time - start_time
    print(f"Inference Time: {inference_time} seconds")

def create_matrix(M):
    import math
    return torch.tensor([math.cos(math.pi * i / (2 * M)) for i in range(1, M + 1)], dtype=torch.float)

class MultiheadGQA(nn.Module):
    def __init__(self, embed_dim = 512, query_heads=8, kv_heads=2):
        super().__init__()
        self.query_heads = query_heads
        self.kv_heads = kv_heads
        self.embed_dim = embed_dim
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        kv_embed_dim = embed_dim // query_heads * kv_heads
        self.k_proj = nn.Linear(embed_dim, kv_embed_dim, bias=False)
        self.v_proj = nn.Linear(embed_dim, kv_embed_dim, bias=False)
        self.out_proj = nn.Linear(kv_embed_dim, embed_dim, bias=False)

    def forward(self, x):
        b, h, w, self.embed_dim = x.shape

        x = x.reshape(1, h*w, -1)
        # query = torch.randn(1, h*w, embed_dim, device="cuda", dtype=torch.float16)
        # key = torch.randn(1, int((h/2)*(w/2)), self.embed_dim)
        # value = torch.randn(1, int((h/2)*(w/2)), self.embed_dim)
        
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        q = q.view(1, h*w, self.query_heads, -1).permute(0, 2, 1, 3)
        k = k.view(1, h*w, self.kv_heads, -1).permute(0, 2, 1, 3)
        v = v.view(1, h*w, self.kv_heads, -1).permute(0, 2, 1, 3)

        scale = q.size(-1) ** 0.5
        q = q / scale

        num_head_groups = self.query_heads // self.kv_heads
        q = rearrange(q, "b (h g) n d -> b g h n d", g=num_head_groups)

        # einsum 대신 matmul과 view를 사용하여 attention 계산
        similarity = torch.einsum("b g h n d, b h s d -> b h n s", q, k)
        attention = F.softmax(similarity, dim=-1)  # [b, num_head_groups, kv_heads, h*w, (h/2)*(w/2)]
        out = torch.matmul(attention, v)  # [b, self.kv_heads, h*w, embed_dim]

        out = out.view(1, h*w, self.kv_heads, -1)
        out = out.view(1, h*w, -1)

        out = self.out_proj(out)
        return out

def create_distance_matrix(size, x, y):
    # x, y 좌표를 위한 벡터 생성 및 GPU로 이동 (GPU 사용 가능한 경우)
    x_coord = torch.arange(size)
    y_coord = torch.arange(size)

    # x, y 좌표의 차이를 계산하여 중심점 (x, y)로부터의 거리 계산
    distance_matrix = torch.sqrt((x_coord - x).pow(2).unsqueeze(0) + (y_coord - y).pow(2).unsqueeze(1))

    # 거리 행렬을 최대 거리 값으로 정규화
    normalized_distance_matrix = distance_matrix / distance_matrix.max()

    # 정규화된 거리 행렬에 대해 cos 변환 수행
    return torch.cos((torch.pi / 2) * normalized_distance_matrix)

def generate_decay_matrix(height, width):
    # 감쇠 행렬을 0으로 초기화 및 GPU로 이동 (GPU 사용 가능한 경우)
    decay_matrix = torch.zeros(height * width, height * width)

    # 각 중심점에 대해 거리 행렬 생성 및 감쇠 행렬 업데이트
    for i in range(height):
        for j in range(width):
            decay_matrix[i * width + j] = create_distance_matrix(height, i, j).view(-1)

    return decay_matrix


if __name__ == '__main__':
    embed_dim = 256
    num_head = 8
    h, w = 128, 128
    input_tensor = torch.FloatTensor(1,h,w,embed_dim) #Batch size = 1, H = 16, W = 16, dim = 256

    h, w = 56, 56

    import time
    start_time = time.time()
    decay_matrix = generate_decay_matrix(h) #input param: height
    # decay_matrix = generate_decay_matrix(h, w) #input param: height
    end_time = time.time()
    print(f"모듈 실행 시간: {end_time - start_time}초")

    #0. MultiHead GQA
    # embed_dim = 512
    attn_module = MultiheadGQA(embed_dim=embed_dim, query_heads=num_head, kv_heads=2) 
    # attn_module = MultiheadGQA(embed_dim=embed_dim, query_heads=num_head, kv_heads=2, device="cuda", dtype=torch.float16) 
    # shapes: (batch_size, seq_len, embed_dim)
    out = attn_module(input_tensor)
    print(attn_module)
    attn_module.eval()
    flops = FlopCountAnalysis(attn_module, input_tensor)
    print(f'Multi-GQA Module')
    print(flop_count_table(flops))


    out = create_matrix(h)
    #1. (Conventional) Self-attention
    attn_module = attention(embed_dim, num_head, h, w)
    print(attn_module)
    # input_data = [torch.randn(1, h, w, 256)]
    # ops_info_provider = OpsInfoProvider(attn_module, input_tensor)
    # measure_inf_time(attn_module, input_tensor)
    attn_module.eval()
    flops = FlopCountAnalysis(attn_module, input_tensor)
    print(f'Self-Attention Module')
    print(flop_count_table(flops))
    torch.onnx.export(attn_module,input_tensor,"attn.onnx")
    #2. Linear attention
    attn_module = linear_attention(embed_dim, num_head, h, w)
    print(attn_module)
    measure_inf_time(attn_module, input_tensor)
    # attn_module.eval()
    # flops = FlopCountAnalysis(attn_module, input_tensor)
    # print(f'Linear Attention Module')
    # print(flop_count_table(flops))

    #3. Retention
    attn_module = retention(embed_dim, num_head, h, w)
    print(attn_module)
    measure_inf_time(attn_module, input_tensor)
    # attn_module.eval()
    # flops = FlopCountAnalysis(attn_module, input_tensor)
    # print(f'Retention Module')
    # print(flop_count_table(flops))

    #4. Linear Retention
    attn_module = linear_retention(embed_dim, num_head, h, w)
    print(attn_module)
    measure_inf_time(attn_module, input_tensor)
    # attn_module.eval()
    # flops = FlopCountAnalysis(attn_module, input_tensor)
    # print(f'Linear Retention Module')
    # print(flop_count_table(flops))
