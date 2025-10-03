#pragma once
// gradient_mean_filter.cpp  —— C++17
#include <cmath>
#include <deque>
#include <iomanip>
#include <iostream>
#include <vector>

class GradientMeanFilter {
   public:
    // 构造：全参数 or 用默认值
    GradientMeanFilter(int window = 3, double lr = 0.1, int epochs = 50, double thresh = 1.5) {
        reset();
    }

    // —— reset —— //
    // 1) 无参：恢复默认 + 清空历史
    // 2) 带参：一次性把超参 & 历史都重置
    void reset() {
        hist_.clear();
        filt_.clear();
    }

    // —— push —— //
    // 喂一个新样本 x，返回该点的滤波结果
    double push(double x) {
        hist_.push_back(x);
        recompute();          // 重新跑一次完整滤波
        return filt_.back();  // 最新滤波值
    }

    // —— get —— //
    // 1) getFiltered()  -> 整段滤波后的序列（引用，别改！）
    // 2) getLast()      -> 最近一次滤波值
    const std::vector<double>& getFiltered() const { return filt_; }
    double getLast() const { return filt_.empty() ? NAN : filt_.back(); }

    // —— 一次性离线滤波便捷函数（与旧版兼容）—— //
    std::vector<double> filter(const std::vector<double>& data) const {
        GradientMeanFilter tmp(*this);  // 拷贝当前参数
        tmp.hist_ = data;
        tmp.recompute();
        return tmp.filt_;
    }

   private:
    // =========== 关键内部实现 =========== //
    void recompute() {
        if (hist_.empty()) return;
        // 1. 梯度下降估计均值 μ
        double mu = estimateMean(hist_);

        // 2. 估计标准差 σ
        double sigma = estimateStd(hist_, mu);

        // 3. 标记极端值
        std::vector<char> is_out(hist_.size(), 0);
        for (size_t i = 0; i < hist_.size(); ++i)
            is_out[i] = std::fabs(hist_[i] - mu) > thresh_ * sigma;

        // 4. 滑窗均值（排除极端值）
        filt_.assign(hist_.size(), 0.0);
        int half = win_ / 2;
        for (size_t i = 0; i < hist_.size(); ++i) {
            double sum = 0.0;
            int cnt = 0;
            for (int j = int(i) - half; j <= int(i) + half; ++j) {
                if (j < 0 || j >= int(hist_.size()) || is_out[j]) continue;
                sum += hist_[j];
                ++cnt;
            }
            filt_[i] = cnt ? sum / cnt : hist_[i];
        }
    }

    // —— 梯度下降计算均值 —— //
    double estimateMean(const std::vector<double>& data) const {
        double mu = 0.0;
        for (double v : data) mu += v;
        mu /= data.size();  // 初始 = 普通均值

        for (int e = 0; e < epochs_; ++e) {
            double grad = 0.0;
            for (double v : data) grad += (mu - v);
            grad *= (2.0 / data.size());
            mu -= lr_ * grad;
        }
        return mu;
    }

    // —— 标准差（样本方差） —— //
    static double estimateStd(const std::vector<double>& data, double mu) {
        double var = 0.0;
        for (double v : data) var += (v - mu) * (v - mu);
        return std::sqrt(var / data.size());
    }

    // -------- 成员变量 -------- //
    int win_{3};
    double lr_{0.1};
    int epochs_{50};
    double thresh_{1.5};

    std::vector<double> hist_;  // 原始序列
    std::vector<double> filt_;  // 滤波后序列
};
