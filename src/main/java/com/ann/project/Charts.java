package com.ann.project;

import javafx.application.Application;
import javafx.application.Platform;
import javafx.collections.*;
import javafx.scene.Scene;
import javafx.scene.chart.*;
import javafx.scene.control.ListView;
import javafx.scene.layout.HBox;
import javafx.scene.layout.Pane;
import javafx.scene.layout.Priority;
import javafx.scene.layout.VBox;
import javafx.stage.Stage;

import java.io.IOException;
import java.util.*;
import java.util.concurrent.*;



public class Charts extends Application {
    public static void main(String[] args) {
        launch(args);
    }

    @Override
    public void start(Stage stage){
        int seedCount = 10;

        NumberAxis valLossX = new NumberAxis();
        valLossX.setLabel("Epoch");
        NumberAxis valLossY = new NumberAxis();
        valLossY.setLabel("Loss");
        LineChart<Number, Number> valLossChart = new LineChart<>(valLossX, valLossY);
        valLossChart.setTitle("Validation loss");

        NumberAxis traLossX = new NumberAxis();
        valLossX.setLabel("Epoch");
        NumberAxis traLossY = new NumberAxis();
        valLossY.setLabel("Loss");
        LineChart<Number, Number> traLossChart = new LineChart<>(traLossX, traLossY);
        valLossChart.setTitle("Training loss");

        NumberAxis accX = new NumberAxis();
        accX.setLabel("Epoch");
        NumberAxis accY = new NumberAxis();
        accY.setLabel("Accuracy %");
        LineChart<Number, Number> accChart = new LineChart<>(accX, accY);
        accChart.setTitle("Accuracy");


        List<XYChart.Series<Number, Number>> valLossList = new ArrayList<>();
        List<XYChart.Series<Number, Number>> accList = new ArrayList<>();
        List<XYChart.Series<Number, Number>> traLossList = new ArrayList<>();


        for (int i = 0; i < seedCount; i++) {
            XYChart.Series<Number, Number> vls = new XYChart.Series<>();
            vls.setName("Seed " + i);
            valLossChart.getData().add(vls);
            valLossList.add(vls);

            XYChart.Series<Number, Number> as = new XYChart.Series<>();
            as.setName("Seed " + i);
            accChart.getData().add(as);
            accList.add(as);

            XYChart.Series<Number, Number> tls = new XYChart.Series<>();
            tls.setName("Seed " + i);
            traLossChart.getData().add(tls);
            traLossList.add(tls);
        }

        ObservableList<String> finalResults = FXCollections.observableArrayList();
        ListView<String> finalResultsView = new ListView<>(finalResults);

        List<ObservableList<String>> results = new ArrayList<>();
        HBox resultBox = new HBox();
        for (int i = 0; i < seedCount; i++) {
            ObservableList<String> r = FXCollections.observableArrayList();
            results.add(r);
            ListView lv = new ListView<>(r);
            HBox.setHgrow(lv, Priority.ALWAYS);
            resultBox.getChildren().add(lv);
        }

        HBox.setHgrow(valLossChart, Priority.ALWAYS);
        HBox.setHgrow(traLossChart, Priority.ALWAYS);
        stage.setScene(new Scene(new VBox( new HBox(valLossChart, traLossChart), accChart, resultBox,finalResultsView), 1200, 900));
        stage.setTitle("Neural Network Training");
        stage.show();

        double[] finalTrainLoss = new double[seedCount];
        double[] finalValLoss   = new double[seedCount];
        double[] finalAccuracy  = new double[seedCount];

        int cores = Runtime.getRuntime().availableProcessors();
        ExecutorService pool = Executors.newFixedThreadPool(Math.min(cores, seedCount),
                r -> {Thread t = new Thread(r);
                t.setDaemon(true);
                return t;
        });
        CountDownLatch latch = new CountDownLatch(seedCount);

        Thread manager = new Thread(() -> {
            try {
                double[][] images    = MnistLoader.loadImages(MnistLoader.TRAIN_IMAGE_FILE);
                double[][] labels    = MnistLoader.loadLabels(MnistLoader.TRAIN_LABEL_FILE);
                double[][] valImages = MnistLoader.loadImages(MnistLoader.TEST_IMAGE_FILE);
                double[][] valLabels = MnistLoader.loadLabels(MnistLoader.TEST_LABEL_FILE);
                double[][][] data = {images, labels, valImages, valLabels};

                /*

                testing model with.
                single hidden layer: 280
                triple hidden layer: 28, 28, 28
                triple hidden layer: 280, 280, 280
                triple hidden layer: 280, 280, 280 sigmoid
                triple hidden layer descending: 350, 100, 28



                 */

                int[] hiddenLayers = {300,300,300};
                Activation[] activations = {null, new ReLU(),new ReLU(),new ReLU(), new SoftMax()};

                for (int s = 0; s < seedCount; s++) {
                    final int seed = s;
                    pool.submit(() -> {
                        try {
                            NeuralNetwork nn = new NeuralNetwork(
                                    seed, 30, 30, 0.01, activations, hiddenLayers, data, "seed" + seed + ".jClass", false);
                            nn.trainEpoch((epoch, tLoss, vLoss, acc) -> {
                                finalTrainLoss[seed] = tLoss;
                                finalValLoss[seed]   = vLoss;
                                finalAccuracy[seed]  = acc;
                                Platform.runLater(() -> {
                                    valLossList.get(seed).getData()
                                            .add(new XYChart.Data<>(epoch, vLoss));
                                    traLossList.get(seed).getData()
                                            .add(new XYChart.Data<>(epoch, tLoss));
                                    accList.get(seed).getData()
                                            .add(new XYChart.Data<>(epoch, acc));
                                    results.get(seed).add("Seed " + seed + " | epoch " + epoch + " " +
                                            "| train " + String.format("%.3f", tLoss) + " " +
                                            "| val " + String.format("%.3f", vLoss) + " " +
                                            "| acc " + acc + "%");
                                });
                            });
                        } catch (Exception e) {
                            e.printStackTrace();
                        } finally {
                            latch.countDown();
                        }
                    });
                }

                latch.await();

                double meanAcc = Arrays.stream(finalAccuracy).average().orElse(0);
                double minAcc  = Arrays.stream(finalAccuracy).min().orElse(0);
                double maxAcc  = Arrays.stream(finalAccuracy).max().orElse(0);
                double meanVal = Arrays.stream(finalValLoss).average().orElse(0);
                double minVal  = Arrays.stream(finalValLoss).min().orElse(0);
                double maxVal  = Arrays.stream(finalValLoss).max().orElse(0);
                double meanTra = Arrays.stream(finalTrainLoss).average().orElse(0);
                double minTra  = Arrays.stream(finalTrainLoss).min().orElse(0);
                double maxTra  = Arrays.stream(finalTrainLoss).max().orElse(0);

                Platform.runLater(() -> finalResults.add(String.format(
                        """
                                SUMMARY:\s
                                | mean acc %.1f%% (min %.1f, max %.1f) |\
                                
                                | mean val loss %.3f (min %.3f, max %.3f) |\
                                
                                | mean tra loss %.3f (min %.3f, max %.3f) |
                                """,
                        meanAcc, minAcc, maxAcc, meanVal, minVal, maxVal, meanTra, minTra, maxTra)));
            } catch (Exception e) {
                e.printStackTrace();
            } finally {
                pool.shutdown();
            }
        });
        manager.setDaemon(true);
        manager.start();
    }
}
