package com.ann.project;

public interface Activation {
    double[][] apply(double[][] Z);
    double[][] derivative(double[][] A);
}
