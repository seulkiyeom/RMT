from fvcore.nn import FlopCountAnalysis
from fvcore.nn import flop_count_table
import torch
import torch.nn as nn
import numpy as np
import time
from AutoOpInspect import OpsInfoProvider


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


if __name__ == '__main__':
    embed_dim = 256
    num_head = 8
    h, w = 128, 128
    input_tensor = torch.FloatTensor(1,h,w,embed_dim) #Batch size = 1, H = 16, W = 16, dim = 256

    #1. (Conventional) Self-attention
    attn_module = attention(embed_dim, num_head, h, w)
    print(attn_module)
    # input_data = [torch.randn(1, h, w, 256)]
    # ops_info_provider = OpsInfoProvider(attn_module, input_tensor)
    # measure_inf_time(attn_module, input_tensor)
    # attn_module.eval()
    # flops = FlopCountAnalysis(attn_module, input_tensor)
    # print(f'Self-Attention Module')
    # print(flop_count_table(flops))
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
