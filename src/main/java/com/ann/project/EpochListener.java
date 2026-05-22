package com.ann.project;

public interface EpochListener {
    void onEpoch(int epoch, double trainLoss, double valLoss, int accuracy);
}
