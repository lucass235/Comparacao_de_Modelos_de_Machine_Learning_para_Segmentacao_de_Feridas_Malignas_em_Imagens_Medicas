import tensorflow as tf
from tensorflow.keras import layers, models

# ---------------------------------------------------------------------
# Definição do módulo Parallel scSE (P-scSE) como camada customizada Keras
# ---------------------------------------------------------------------
class PscSE(layers.Layer):
    def __init__(self, in_channels, reduction=16, **kwargs):
        super(PscSE, self).__init__(**kwargs)
        self.in_channels = in_channels
        self.reduction = reduction

        # Ramo cSE: pooling global seguido de convoluções 1x1
        self.global_avg_pool = layers.GlobalAveragePooling2D()
        self.conv1 = layers.Conv2D(in_channels // reduction, (1, 1), activation='relu')
        self.conv2 = layers.Conv2D(in_channels, (1, 1), activation='sigmoid')
        
        # Ramo sSE: convolução 1x1 para gerar mapa espacial de atenção
        self.sse_conv = layers.Conv2D(1, (1, 1), activation='sigmoid')

    def call(self, inputs):
        # Ramo cSE
        cse = self.global_avg_pool(inputs)                      # (batch, channels)
        cse = tf.reshape(cse, (-1, 1, 1, self.in_channels))       # (batch, 1, 1, channels)
        cse = self.conv1(cse)
        cse = self.conv2(cse)
        cse_out = inputs * cse
        
        # Ramo sSE
        sse = self.sse_conv(inputs)
        sse_out = inputs * sse
        
        # Combinação: soma (aditiva) e max-out, seguidos de média
        out_add = cse_out + sse_out
        out_max = tf.maximum(cse_out, sse_out)
        out = (out_add + out_max) / 2.0
        return out

# ---------------------------------------------------------------------
# Classe FUSegNet adaptada para ser chamada conforme:
# if (modelName == "fusegnet"):
#     fusegnet = FUSegNet(n_filters=n_filters, input_dim_x=input_dim_x, input_dim_y=input_dim_y, num_channels=3)
#     model, model_name = fusegnet.get_FUSegNet()
#     print("Modelo FUSegNet Carregado!")
# ---------------------------------------------------------------------
class FUSegNet:
    def __init__(self, n_filters, input_dim_x, input_dim_y, num_channels=3, num_classes=1):
        self.input_dim_x = input_dim_x
        self.input_dim_y = input_dim_y
        self.num_channels = num_channels
        self.num_classes = num_classes
        self.n_filters = n_filters  # Base para escala dos filtros
        self.model = None
        self.model_name = "FUSegNet"
    
    def build_model(self):
        input_shape = (self.input_dim_x, self.input_dim_y, self.num_channels)
        inputs = layers.Input(shape=input_shape)
        
        # Encoder: EfficientNetB7 pré-treinado (sem a cabeça final)
        encoder = tf.keras.applications.EfficientNetB7(
            include_top=False,
            weights='imagenet',
            input_tensor=inputs
        )
        
        # Atualização dos skip connections: usamos camadas com resoluções compatíveis
        # Para um input de 224x224, é comum usar:
        #  - "block4a_activation" com shape (14,14,?)
        #  - "block3a_activation" com shape (28,28,?)
        #  - "block2a_activation" com shape (56,56,?)
        skip_names = ['block4a_activation', 'block3a_activation', 'block2a_activation']
        skips = [encoder.get_layer(name).output for name in skip_names]
        encoder_output = encoder.output  # Geralmente (7,7,2560)
        
        # Decodificador
        # Bloco 1: upsample de (7,7) para (14,14) e concatena com "block4a_activation"
        x = layers.Conv2DTranspose(self.n_filters * 16, (2, 2), strides=(2, 2), padding='same')(encoder_output)
        x = layers.concatenate([x, skips[0]])
        x = layers.Conv2D(self.n_filters * 16, (3, 3), activation='relu', padding='same')(x)
        x = layers.Conv2D(self.n_filters * 16, (3, 3), activation='relu', padding='same')(x)
        x = PscSE(self.n_filters * 16)(x)
        
        # Bloco 2: upsample de (14,14) para (28,28) e concatena com "block3a_activation"
        x = layers.Conv2DTranspose(self.n_filters * 8, (2, 2), strides=(2, 2), padding='same')(x)
        x = layers.concatenate([x, skips[1]])
        x = layers.Conv2D(self.n_filters * 8, (3, 3), activation='relu', padding='same')(x)
        x = layers.Conv2D(self.n_filters * 8, (3, 3), activation='relu', padding='same')(x)
        x = PscSE(self.n_filters * 8)(x)
        
        # Bloco 3: upsample de (28,28) para (56,56) e concatena com "block2a_activation"
        x = layers.Conv2DTranspose(self.n_filters * 4, (2, 2), strides=(2, 2), padding='same')(x)
        x = layers.concatenate([x, skips[2]])
        x = layers.Conv2D(self.n_filters * 4, (3, 3), activation='relu', padding='same')(x)
        x = layers.Conv2D(self.n_filters * 4, (3, 3), activation='relu', padding='same')(x)
        x = PscSE(self.n_filters * 4)(x)
        
        # Bloco 4: upsample de (56,56) para (112,112)
        x = layers.Conv2DTranspose(self.n_filters * 2, (2, 2), strides=(2, 2), padding='same')(x)
        x = layers.Conv2D(self.n_filters * 2, (3, 3), activation='relu', padding='same')(x)
        x = layers.Conv2D(self.n_filters * 2, (3, 3), activation='relu', padding='same')(x)
        x = PscSE(self.n_filters * 2)(x)
        
        # Bloco 5: upsample de (112,112) para (224,224)
        x = layers.Conv2DTranspose(self.n_filters, (2, 2), strides=(2, 2), padding='same')(x)
        x = layers.Conv2D(self.n_filters, (3, 3), activation='relu', padding='same')(x)
        x = layers.Conv2D(self.n_filters, (3, 3), activation='relu', padding='same')(x)
        x = PscSE(self.n_filters)(x)
        
        # Camada final: convolução 1x1 com ativação Sigmoid para segmentação binária
        outputs = layers.Conv2D(self.num_classes, (1, 1), activation='sigmoid')(x)
        
        model = models.Model(inputs, outputs)
        return model
    
    def get_FUSegNet(self):
        if self.model is None:
            self.model = self.build_model()
        return self.model, self.model_name

# ---------------------------------------------------------------------
# Exemplo de chamada conforme:
# if (modelName == "fusegnet"):
#     fusegnet = FUSegNet(n_filters=n_filters, input_dim_x=input_dim_x, input_dim_y=input_dim_y, num_channels=3)
#     model, model_name = fusegnet.get_FUSegNet()
#     print("Modelo FUSegNet Carregado!")
# ---------------------------------------------------------------------
if __name__ == '__main__':
    modelName = "fusegnet"
    n_filters = 32
    input_dim_x = 224
    input_dim_y = 224

    if modelName == "fusegnet":
        fusegnet = FUSegNet(n_filters=n_filters, input_dim_x=input_dim_x, input_dim_y=input_dim_y, num_channels=3)
        model, model_name = fusegnet.get_FUSegNet()
        print("Modelo FUSegNet Carregado!")
        model.summary()
