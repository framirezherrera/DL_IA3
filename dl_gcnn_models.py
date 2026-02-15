# ----------------------------------------------------------------------------------------------------------------------------------
# Archivo: dl_gcnn_models.py
# Descripción: Clases que implementan los modelos basados en arquitecturas Gated CNN para Seq-to-Seq
# Implementado por: Felipe Ramírez Herrera (basado en código de terceros)
# Curso Aprendizaje Profundo 1 y 2. Universidad de Valencia / ADEIT
# Ultima revisión: 11/04/2024 
# ----------------------------------------------------------------------------------------------------------------------------------

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import math

# Esta función genera una matriz de tensores que representan las posiciones en una secuencia. 
# Es una función auxiliar utilizada para crear las embeddings posicionales necesarias para el modelo
        
def create_postional_tensor(length, batch_size, device): 
    return torch.arange(0, length, device=device).unsqueeze(0).repeat(batch_size, 1) # [0, 1, 2, 3, ..., length - 1] 

# CLASE: GatedConvEncoder
# Esta es una clase que define un módulo de encoder para una arquitectura Encoder - Decoder que utiliza convoluciones. 
# Basado en convoluciones con compuertas que utiliza estas embeddings posicionales y realiza convoluciones, activaciones GLU y conexiones 
# residuales para procesar la entrada y generar salidas.
# Based on https://github.com/facebookresearch/fairseq/blob/main/fairseq/

class GatedConvEncoder(nn.Module):
    def __init__(self, input_dim, emb_dim, hid_dim, n_layers, kernel_size, dropout, device, max_length = 100):
        super().__init__()
        assert kernel_size % 2 == 1, "Tamaño de kernel es inválido"
        self.scale = math.sqrt(0.5) 
        self.tok_embedding = nn.Embedding(input_dim, emb_dim)   # Capa de embedding para tokens
        self.pos_embedding = nn.Embedding(max_length, emb_dim)  # Capa de embedding posicional
        self.incoming_projection = nn.Linear(emb_dim, hid_dim)  # Proyección de entrada
        self.outgoing_projection = nn.Linear(hid_dim, emb_dim)  # Proyección de salida
        self.convs = nn.ModuleList([nn.Conv1d(in_channels = hid_dim, out_channels = 2 * hid_dim, kernel_size = kernel_size, padding = (kernel_size - 1) // 2) for _ in range(n_layers)])
        self.dropout = nn.Dropout(dropout)
        self.device = device

    # x = [batch size, x len]  
    def forward(self, x):        
        batch_size, src_len = x.shape[0], x.shape[1]   
        pos = create_postional_tensor(src_len, batch_size, self.device) # [batch size, x len]  
        # Calcula las embeddings de los tokens y las embeddings posicionales.
        te = self.tok_embedding(x)      # [BS, X, emb dim]
        pe = self.pos_embedding(pos)    # [BS, X, emb dim]
        # Combina las embeddings sumándolas.
        embedded = self.dropout(te + pe)                # [BS, X, emb dim]
        # Proyecta las embeddings combinadas a una dimensión oculta.
        conv_input = self.incoming_projection(embedded) # [BS, X, hid dim]
        # Aplica capas de convolución, activación GLU y conexiones residuales.
        conv_input = conv_input.permute(0, 2, 1)        # [BS, hid dim, X]  
        for conv in self.convs:
            # Paso a través de la capa convolucional
            conved = conv(self.dropout(conv_input))         # [BS, 2 * hid dim, X]
            # Paso a través de la función de activación GLU, GLU es más estable que ReLU y aprende más rápido que sigmoid.
            conved = F.glu(conved, dim = 1)                 # [batch size, hid dim, src len]            
             # Aplica conexión residual
            conved = (conved + conv_input) * self.scale     # [batch size, hid dim, src len]
            # Prepara para la próxima iteración
            conv_input = conved       
        # Permuta y convierte de nuevo a emb dim
        conved = self.outgoing_projection(conved.permute(0, 2, 1))  # [batch size, src len, emb dim]
        combined = (conved + embedded) * self.scale                 # combined = [batch size, src len, emb dim]
        return conved, combined

# Este código implementa un decodificador utilizando convoluciones con compuertas (gated convolutions) y mecanismo de atención 
# para generar secuencias de salida basadas en una secuencia de entrada codificada. 
# Se agrega funcionalidad de weight-sharing entre las capas de tok_embedding y fc_out (propuesto alumno)
# Based on https://github.com/facebookresearch/fairseq/blob/main/fairseq/

class GatedConvDecoder(nn.Module):
    def __init__(self, output_dim, emb_dim, hid_dim, n_layers, kernel_size, dropout, trg_pad_idx, device, max_length = 100):
        super().__init__()   
        self.kernel_size = kernel_size
        self.trg_pad_idx = trg_pad_idx
        self.device = device
        self.scale = self.scale = math.sqrt(0.5)  
        self.tok_embedding = nn.Embedding(output_dim, emb_dim)          # Embedding para tokens de salida
        self.pos_embedding = nn.Embedding(max_length, emb_dim)          # Embedding posicional
        # Proyección desde la salida de la convolución hasta la dimensión de embedding
        self.dec_incoming_projection = nn.Linear(emb_dim, hid_dim)
        # Proyección desde la dimensión de embedding hasta el tamaño de la convolución
        self.dec_outgoing_projection = nn.Linear(hid_dim, emb_dim)       
        # Proyección desde la salida de la convolución hasta la dimensión de embedding (para la atención)
        self.attn_incoming_projection = nn.Linear(hid_dim, emb_dim)
        # Proyección desde la dimensión de embedding hasta el tamaño de la convolución (para la atención)
        self.attn_outgoing_projection = nn.Linear(emb_dim, hid_dim) 
        # Proyección de vuelta al tamaño del vocabulario 
        self.fc_out = nn.Linear(emb_dim, output_dim)
        # Convoluciones temporales
        self.convs = nn.ModuleList([nn.Conv1d(in_channels = hid_dim, out_channels = 2 * hid_dim, kernel_size = kernel_size) for _ in range(n_layers)])
        self.dropout = nn.Dropout(dropout)

    def shared_weights_config(self):
        self.fc_out.weight = self.tok_embedding.weight

    # Calcula la atención
    # embedded          [batch size, trg len, emb dim]
    # conved            [batch size, hid dim, trg len]
    # encoder_conved    [batch size, src len, emb dim]
    # encoder_combined  [batch size, src len, emb dim]
        
    def calculate_attention(self, embedded, conved, encoder_conved, encoder_combined):
        # Proyección y combinación de embeddings
        conved_emb = self.attn_incoming_projection(conved.permute(0, 2, 1))             # [batch size, trg len, emb dim]
        combined = (conved_emb + embedded) * self.scale                                 # [batch size, trg len, emb dim]    
        permuted = encoder_conved.permute(0, 2, 1)
        energy = torch.matmul(combined, permuted)                # [batch size, trg len, src len]
        # Softmax para obtener los pesos de atención
        attention = F.softmax(energy, dim=2)                                            # [batch size, trg len, src len]
        attended_encoding = torch.matmul(attention, encoder_combined)                   # [batch size, trg len, emd dim]
        # Conversión de emb dim a hid dim
        attended_encoding = self.attn_outgoing_projection(attended_encoding)            # [batch size, trg len, hid dim]
        # Aplicación de conexión residual
        attended_combined = (conved + attended_encoding.permute(0, 2, 1)) * self.scale  # [batch size, hid dim, trg len]
        return attention, attended_combined


    # Realiza el proceso de decodificación
    # trg               [trg len, batch size]
    # encoder_conved    [batch size, src len, emb dim]
    # encoder_combined  [batch size, src len, emb dim]

    def forward(self, trg, encoder_conved, encoder_combined):              
        batch_size, trg_len = trg.shape[0], trg.shape[1]   
        # Creación del tensor de posiciones
        pos = create_postional_tensor(trg_len, batch_size, self.device)                         # [trg len, batch size]
        # Embedding de tokens y posiciones
        tok_embedded = self.tok_embedding(trg)                                                  # [trg len, batch size, emb dim]
        pos_embedded = self.pos_embedding(pos)                                                  # [trg len, batch size, emb dim]
        embedded = self.dropout(tok_embedded + pos_embedded)                                    # [batch size, trg len, emb dim]
        # Proyección al tamaño de la convolución
        conv_input = self.dec_incoming_projection(embedded)                                     # [batch size, trg len, hid dim]      
        # Permutación para la capa convolucional
        conv_input = conv_input.permute(0, 2, 1)                                                # [batch size, hid dim, trg len]
        batch_size = conv_input.shape[0]
        hid_dim = conv_input.shape[1]
        for conv in self.convs:
            conv_input = self.dropout(conv_input)
            padding = torch.zeros(batch_size, hid_dim, self.kernel_size - 1, device=self.device).fill_(self.trg_pad_idx)   
            padded_conv_input = torch.cat((padding, conv_input), dim = 2) # [batch size, hid dim, trg len + kernel size - 1]
            conved = conv(padded_conv_input)                              # [batch size, 2 * hid dim, trg len]
            conved = F.glu(conved, dim = 1)                               # [batch size, hid dim, trg len]
            attention, conved = self.calculate_attention(embedded, conved, encoder_conved, encoder_combined)  # [batch size, trg len, src len]
            conved = (conved + conv_input) * self.scale                   # [batch size, hid dim, trg len]
            conv_input = conved          
        conved = self.dec_outgoing_projection(conved.permute(0, 2, 1))      # [batch size, trg len, emb dim]
        output = self.fc_out(self.dropout(conved))                          # [batch size, trg len, output dim]
        return output, attention

# Modelo Single Task
# Este código define una arquitectura de modelo de secuencia a secuencia (Seq2Seq) utilizando convoluciones con compuertas
# (gated convolutions) para el codificador y el decodificador. 

class GatedConvSeq2Seq(nn.Module):
    def __init__(self, name, encoder, decoder):
        """
        Constructor de la clase.

        Parámetros:
         - name (str): Nombre del modelo.
         - encoder (nn.Module): Modelo del codificador.
         - decoder (nn.Module): Modelo del decodificador.
        
        """
        super().__init__()
        self.name = name
        self.encoder = encoder
        self.decoder = decoder
    # x = [batch size, x len]
    # y = [batch size, y len - 1]
    def forward(self, x, y):
        """
        Método de propagación hacia adelante del modelo.

        Parámetros:
         - x (Tensor): Secuencia de entrada. Dimensiones: [batch size, x len].
         - y (Tensor): Secuencia de salida. Dimensiones: [batch size, y len - 1].

        Retorna:
         - output (Tensor): Salida del decodificador. Dimensiones: [batch size, trg len - 1, output dim].
         - attention (Tensor): Pesos de atención. Dimensiones: [batch size, trg len - 1, src len].
        """
        encoder_conved, encoder_combined = self.encoder(x)
        output, attention = self.decoder(y, encoder_conved, encoder_combined) # [batch size, trg len - 1, output dim], [batch size, trg len - 1, src len]        
        return output, attention
    
    def encode(self, x):
        """
        Método para codificar una secuencia de entrada.

        Parámetros:
         - x (Tensor): Secuencia de entrada. Dimensiones: [batch size, x len].

        Retorna:
         - encoder_conved (Tensor): Salida del codificador. Dimensiones: [batch size, src len, emb dim].
         - encoder_combined (Tensor): Salida combinada del codificador. Dimensiones: [batch size, src len, emb dim].
        """
        return self.encoder(x)

    def decode(self, y, latent, combined):
        """
        Método para decodificar una secuencia de salida.

        Parámetros:
         - y (Tensor): Secuencia de salida. Dimensiones: [batch size, y len - 1].
         - latent (Tensor): Salida del codificador. Dimensiones: [batch size, src len, emb dim].
         - combined (Tensor): Salida combinada del codificador. Dimensiones: [batch size, src len, emb dim].

        Retorna:
         - output (Tensor): Salida del decodificador. Dimensiones: [batch size, trg len - 1, output dim].
         - attention (Tensor): Pesos de atención. Dimensiones: [batch size, trg len - 1, src len].
        """
        return self.decoder(y, latent, combined) 

    def getName(self):
        """
        Método para obtener el nombre del modelo.

        Retorna:
         - name (str): Nombre del modelo.
        """
        return self.name

# Model Double Task con opción de Weight-Sharing en ambos decodificadores para reducir la firma de memoria.
# Este código define una arquitectura de modelo de secuencia a secuencia (Seq2Seq) utilizando convoluciones con compuertas
# (gated convolutions) para el codificador y dos decodificadores (Tarea A y Tarea B). 

class GatedConvDualTaskSeq2Seq(nn.Module):
    def __init__(self, name : str, encoder : GatedConvEncoder, decoder_a : GatedConvDecoder, decoder_b : GatedConvDecoder, shared_weights = False):
        """
        Constructor de la clase.

        Parámetros:
         - name (str): Nombre del modelo.
         - encoder (GatedConvEncoder): Modelo del codificador.
         - decoder_a (GatedConvDecoder): Modelo del decodificador para la tarea A.
         - decoder_b (GatedConvDecoder): Modelo del decodificador para la tarea B.
         - shared_weights (bool): Indica si se deben compartir los pesos de los decodificadores para ambas tareas.
        """
        super().__init__()
        self.name = name
        self.encoder = encoder
        self.decoder_a = decoder_a # Decoder for Task No. 1 (EN to ES)
        self.decoder_b = decoder_b # Decoder for Task No. 2 (EN to FR)

        if (shared_weights):
            self.decoder_a.shared_weights_config()
            self.decoder_b.shared_weights_config()

    # x = [batch size, src len]
    # y0 = [batch size, y0 len - 1] (<bos> token sliced off without <eos>)
    # y1 = [batch size, y1 len - 1] (<bos> token sliced off without <eos>) 
    def forward(self, x, y0, y1, returns_attention = False):       
        """
        Método de propagación hacia adelante del modelo para realizar ambas tareas.

        Parámetros:
         - x (Tensor): Secuencia de entrada. Dimensiones: [batch size, src len].
         - y0 (Tensor): Secuencia de salida para la tarea A. Dimensiones: [batch size, y0 len - 1] (sin el token <bos> pero con <eos>).
         - y1 (Tensor): Secuencia de salida para la tarea B. Dimensiones: [batch size, y1 len - 1] (sin el token <bos> pero con <eos>).
         - returns_attention (bool): Indica si se deben devolver los pesos de atención.

        Retorna:
         - output_a (Tensor): Salida del decodificador para la tarea A. Dimensiones: [batch size, y0 len - 1, output dim].
         - output_b (Tensor): Salida del decodificador para la tarea B. Dimensiones: [batch size, y1 len - 1, output dim].
         - (Opcional) attention_a (Tensor): Pesos de atención para la tarea A. Dimensiones: [batch size, y0 len - 1, src len].
         - (Opcional) attention_b (Tensor): Pesos de atención para la tarea B. Dimensiones: [batch size, y1 len - 1, src len].
        """  
        encoder_conved, encoder_combined = self.encoder(x) # [batch size, src len, emb dim], [batch size, src len, emb dim]
        output_a, attention_a = self.decoder_a(y0, encoder_conved, encoder_combined) # [batch size, y0 len - 1, output dim], attention_a = [batch size, y0 len - 1, x len]
        output_b, attention_b = self.decoder_b(y1, encoder_conved, encoder_combined) # [batch size, y1 len - 1, output dim], attention_b = [batch size, y1 len - 1, x len]
        
        if (returns_attention):
            return output_a, output_b, attention_a, attention_b
        else:
            return output_a, output_b


    def encode(self, x):
        """
        Método para codificar una secuencia de entrada.

        Parámetros:
         - x (Tensor): Secuencia de entrada. Dimensiones: [batch size, src len].

        Retorna:
         - encoder_conved (Tensor): Salida del codificador. Dimensiones: [batch size, src len, emb dim].
         - encoder_combined (Tensor): Salida combinada del codificador. Dimensiones: [batch size, src len, emb dim].
        """
        return self.encoder(x)

    def decode_for_task_a(self, y, latent, combined):
        """
        Método para decodificar una secuencia de salida para la tarea A.

        Parámetros:
         - y (Tensor): Secuencia de salida. Dimensiones: [batch size, y len - 1].
         - latent (Tensor): Salida del codificador. Dimensiones: [batch size, src len, emb dim].
         - combined (Tensor): Salida combinada del codificador. Dimensiones: [batch size, src len, emb dim].

        Retorna:
         - output (Tensor): Salida del decodificador. Dimensiones: [batch size, trg len - 1, output dim].
        """
        return self.decoder_a(y, latent, combined) 

    def decode_for_task_b(self, y, latent, combined):
        """
        Método para decodificar una secuencia de salida para la tarea B.

        Parámetros:
         - y (Tensor): Secuencia de salida. Dimensiones: [batch size, y len - 1].
         - latent (Tensor): Salida del codificador. Dimensiones: [batch size, src len, emb dim].
         - combined (Tensor): Salida combinada del codificador. Dimensiones: [batch size, src len, emb dim].

        Retorna:
         - output (Tensor): Salida del decodificador. Dimensiones: [batch size, trg len - 1, output dim].
        """
        return self.decoder_b(y, latent, combined) 

    def getName(self):
        """
        Método para obtener el nombre del modelo.

        Retorna:
         - name (str): Nombre del modelo.
        """
        return self.name  
