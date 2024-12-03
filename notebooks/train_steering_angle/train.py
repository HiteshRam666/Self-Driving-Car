import os
import tensorflow as tf
import driving_data
from model import build_model

class DataLogger:
    def __init__(self, logs_path):
        self.logs_path = logs_path
        self.summary_writer = tf.summary.create_file_writer(logs_path)

    def log_summary(self, summary, step):
        with self.summary_writer.as_default():
            tf.summary.scalar("loss", summary, step)
            self.summary_writer.flush()
    
    def close(self):
        self.summary_writer.close()

class Trainer:
    def __init__(self, model, log_dir, logger, l2_norm_const = 0.001, learning_rate = 1e-4):
        self.log_dir = log_dir 
        self.l2_norm_const = l2_norm_const
        self.logger = logger

        self.model = model 
        self.loss_object = tf.keras.losses.MeanSquaredError()
        self.optimizer = tf.keras.optimizers.Adam(learning_rate)

        self.checkpoint = tf.train.Checkpoint(model=self.model, optimizer=self.optimizer)
        self.checkpoint_manager = tf.train.CheckpointManager(self.checkpoint, log_dir, max_to_keep=5)

    def train(self, epochs, batch_size):
        for epoch in range(epochs):
            self._train_one_epoch(epoch, batch_size)
            print(f"Epoch {epoch + 1}/{epochs} completed.")
        
    def _train_one_epoch(self, epoch, batch_size):
        for i in range(int(driving_data.num_train_images / batch_size)):
            xs, ys = driving_data.LoadTrainBatch(batch_size)
            self._train_step(xs, ys)

            if i % 10 == 0:
                self._log_progress(epoch, i, batch_size)
            
            if i % batch_size == 0:
                self._save_checkpoint() 

    @tf.function
    def _train_step(self, xs, ys):
        with tf.GradientTape() as tape:
            predictions = self.model(xs, training=True)
            loss_value = self.loss_object(ys, predictions)
            loss_value += self._l2_regularization() 
        
        grads = tape.gradient(loss_value, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))
        return loss_value
    
    def _l2_regularization(self):
        l2_loss = tf.add_n([tf.nn.l2_loss(v) for v in self.model.trainable_variables])
        return self.l2_norm_const * l2_loss

    def _log_progress(self, epoch, step, batch_size):
        xs, ys = driving_data.LoadTrainBatch(batch_size)
        loss = self._train_step(xs, ys)
        self.logger.log_summary(loss, step)

    def _save_checkpoint(self):
        self.checkpoint_manager.save()

if __name__ == "__main__":
    log_dir = './logs'
    model = build_model((66, 200, 3))
    logger = DataLogger(log_dir)
    trainer = Trainer(model, log_dir, logger)
    trainer.train(epochs=10, batch_size=64)
    logger.close()
