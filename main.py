#! /usr/bin/python3 -s
# -*- coding: utf-8 -*-

"""
Nome: Jahir Gilberth Medina Lopez
Cod. USP: 10659682
Diciplina: SCC0251
Ano/Semestre: 2018/1er
Trabalho: Trabalho 2: realce e superresolução
"""

import numpy as np
import imageio as img
import sys

# from matplotlib import pyplot as plt
# import time as tm

# import operator


def multi_histogram_debug(histogram_data, debug):
    """
    
    Funcion simple para debug, recibe los histogramas generados
    para cada imagen y los plotea en una sola escena.
    >Es necesario descomentar la libreria matplotlib
    No es vital para la ejecucion del programa
    """

    if debug:
        fig_amount = len(histogram_data)
        fig_row = fig_amount/2
        fig_col = fig_amount - fig_row
            
        fig = plt.figure()
        fig_list = [None for i in range(fig_amount)]
            
        index = 0
        for sub_histogram_data in histogram_data:
            fig_list[index] = fig.add_subplot(fig_row ,fig_col, index+1)
            fig_list[index].plot([i for i in range(len(sub_histogram_data))] ,sub_histogram_data)
            index += 1

        plt.savefig("./dummyMULT.png")

    else:
        pass

def image_reader(path):
    """
    
    re-adaptacion de la funcion imread del paquete imageio
    disenado para que se puede cargar directamente imagenes en formato PNG

    """
    image_data = img.imread(path + ".png",'png')
    
    return image_data

def interative_image_reader(path, amount_of_images):

    """
    
    modelo iterativo de la funcion imread del paquete imageio
    disenado para que se pueda cargar un grupo de imagenes en formato PNG


    """

    output = []
    
    for i in range(amount_of_images):
        output.append(image_reader(path + str(i+1)))

    return output

def fusion_matrix(matrix_pack):
    """
    
    Funcion disenada para crear la imagen en super-resolucion

    forma de funcionamiento detallado en cada sub-bloque

    """
    [x_dim, y_dim] = list(matrix_pack[0].shape)


    """
    En esta seccion se reinterpreta las imagenes como matrices de numberpy, permitiendo
    se pueda operar con ella usando las funciones de numberpy
    """
    for i in range(4):
        matrix_pack[i] = np.matrix(matrix_pack[i])

    """
    Se crea un sub elemento para el acarreo de datos,
    en este caso es una matriz formada por las columnas de la imagen 1 y la imagen 2 de forma intercalada
    de tal forma que se crea una matriz de Nx(N*2)
    """
    image_carry_1 = matrix_pack[0][:,0]

    for i in range(1, x_dim*2):
        # print(image_carry_1.shape)
        if i%2 is 0:
            image_carry_1 = np.hstack( (image_carry_1, matrix_pack[0][:,int(i/2)]))
        else:
            image_carry_1 = np.hstack( (image_carry_1, matrix_pack[1][:,int((i-1)/2)]))

    # print(image_carry_1.shape)

    """
    Se crea un sub elemento para el acarreo de datos,
    en este caso es una matriz formada por las columnas de la imagen 3 y la imagen 4 de forma intercalada
    de tal forma que se crea una matriz de Nx(N*2)
    """
    image_carry_2 = matrix_pack[2][:,0]

    for i in range(1,x_dim*2):
        # print(image_carry_2.shape)
        # print(image_carry_2)
        if i%2 is 0:
            image_carry_2 = np.hstack( (image_carry_2, matrix_pack[2][:,int(i/2)]))
        else:
            image_carry_2 = np.hstack( (image_carry_2, matrix_pack[3][:,int((i-1)/2)]))

    # print(image_carry_2.shape)

    """
    En esta parte se combinan las dos sub matrices de acarreo generadas , de tal forma que
    el resultado final sea una matriz (N*2)x(N*2) conformado por
    las FILAS del primer acarreo y las del segundo acarreo, intercalados.
    """

    image_carry = image_carry_1[0]

    for i in range(1,y_dim*2):
        if i%2 is 0:
            image_carry = np.vstack( (image_carry, image_carry_1[int(i/2)]))
        else:
            image_carry = np.vstack( (image_carry, image_carry_2[int((i-1)/2)]))

    # print(image_carry.shape)
    """
    Razonamiento:
    La conformacion de cada sub-bloque 2x2 dentro de la imagen en super-resolucion no es mas que los n-simos
    elementos de cada matriz, para emular ello, es tan sencillo como que para la 1er matriz, sus columnas siempre ocuparan
    lugares impares, para la 2da matriz, lugares pares

    En el caso de la matriz 3 y 4 sucede lo mismo pero desplazado una fila hacia abajo.

    Teniendo estas 2 sub-matrices creadas por el intercalado de columnas, solo se procede a intercalar filas, haciendo que
    la combinacion final cumpla con el metodo de superresolucion establecido
    """
    return image_carry

class enhance_image:
    """
    Esta clase se crea con el motivo de aprevechar el tratamiento que da python3.6 a la data dentro de los objetos
    evitando sobrecargas o copias inecesarias de datos
    """
    def __init__(self, low_img, amount_of_images, high_res_img, super_res_ratio, s_param = None, enhance_method=None, debug=False):
        self.amount_of_images = amount_of_images
        
        self.base_name = low_img
        
        self.set_of_low_res_img = interative_image_reader(low_img, self.amount_of_images)
        self.high_res_img = image_reader(high_res_img)

        self.x_dim = int(list(self.high_res_img.shape)[0]/2)
        self.y_dim = int(list(self.high_res_img.shape)[1]/2)

        self.enhance_img_list = None
        self.super_res_ratio = super_res_ratio

        self.s_param = s_param
        self.method_type = enhance_method
        self.debug_status = debug
        """
        Parametros propios de la clase destinado al funcionamiento de la misma

        Explicaciones:

        amount_of_images es el numero de muestras en baja resolucion

        set_of_low_res_img son las imagenes en memoria

        high_res_img es la imagen final con la que se comparara en memoria


        x_dim, y_dim es el tamano de las imagenes pequenas , calculadas respecto al valor de la imagen grande

        el resto son valores utilitarios, explicados dentro de cada funcion
        """


    def generate_histogram(self,image_data=None, histogram_range=range(257), debug=False):
        carry = []

        if image_data is None:
            image_data = self.set_of_low_res_img
        else:
            return None
        
        """
        en el siguiente bloque de codigo, se hace uso de la funcion numpy.histogram para generar
        un histograma de la matriz referente a las imagenes en baja resolucion
        """
        for sub_data in image_data:
            sub_carry = np.histogram(sub_data, histogram_range)
            carry.append(sub_carry[0])

        multi_histogram_debug(carry, debug)

        """
        Se devuelve una lista con los histogramas, cada indice dentro de la lista implica el numero de imagen
        """
        self.histogram_data = carry

    def method_0(self, debug=False):
        pass
        self.enhance_img_list = self.set_of_low_res_img
        """
        Especificado en el PDF:  el metodo 0 no hace nada
        """

    def method_1(self, debug=False):

        self.generate_histogram()

        carry = []
        """
        carry = contenedor para las imagenes mejoradas bajo el metodo en cuestion
        ratio = valor obtenido de la formula de normalizacion ((L-1)/(N*M)) de forma que
        facilite la operacion vectorial al ser aplicado al histograma
        """
        ratio = (255/(self.x_dim*self.y_dim))

        for index in range(self.amount_of_images):
            
            sub_cumulative = np.cumsum(self.histogram_data[index])*ratio

            equalized_image = np.zeros((self.x_dim, self.y_dim), np.uint8)

            """
            en las dos lineas anteriores se aprovecha de los metodos numpy.cumsum y la operacion vectorizada de multiplicacion
            para generar el histograma normalizado por cada imagen

            equalized_imagen es un contendero para obtener la imagen equalizada
            """
            
            for i in range(256):
                
                x =  self.set_of_low_res_img[index] == i
                x = x.astype(np.uint8)

                equalized_image = np.add(equalized_image, np.multiply(x, sub_cumulative.item(i)))
                """
                en estas 3 lineas anterios se realiza la susticion de los valores iniciales por los normalizados,
                primero se crea una matriz binaria donde solo existan 1 cuando el valor coincide con la intesidad a remplazarse
                segundo se convierte a enteres de 8byts
                tercer se multiplica por el escalar respectivo a la intesidad que se remplazara, ya que la multiplicacion es uno a uno
                facilita la sutitucion, finalmente se suma al anterior valor
                ya que no existen superposicion de valores, se tiene la certeza la matriz genera sera la imagen equalizada
                """
            carry.append(equalized_image.astype(np.uint8))

            multi_histogram_debug(carry, debug)

        self.enhance_img_list = carry


    def method_2(self, debug=False):

        self.generate_histogram()

        ratio = (255/(self.x_dim*self.y_dim))

        histogram_acumulator = np.zeros(256, np.uint8)
        """
        Similiar al metodo 1 , diferencia que luego de calular el valor ratio, se debe sumar los histogramas (siguiente for)
        """
        for i in range(self.amount_of_images):
            histogram_acumulator = np.add(histogram_acumulator, self.histogram_data[i])

        histogram_normalized = np.cumsum(histogram_acumulator)*ratio
        """
        luego de sumar se realiza la normalizacion mediante operaciones vectoriales
        """
        carry = []

        for index in range(self.amount_of_images):

            equalized_image = np.zeros((self.x_dim, self.y_dim), np.uint8)
            
            for i in range(256):
                
                x =  self.set_of_low_res_img[index] == i
                x = x.astype(np.uint8)

                equalized_image = np.add(equalized_image, np.multiply(x, histogram_normalized.item(i)))

            """
            Equalizacion de las imagenes en baja resolucion siguiendo la logica del metodo 1
            """

            carry.append(equalized_image.astype(np.uint8))

            multi_histogram_debug(carry, debug)

        self.enhance_img_list = carry

    def method_3(self, debug=False):
        
        carry = []

        """
        La operacion de ajuste de gamma es identica a la del PDF, modificacion: se vectorizo su aplicacion
        """

        for i in range(self.amount_of_images):
            equalized_image = np.multiply(np.power(np.divide(self.set_of_low_res_img[i], 255), (1/self.s_param)), 255).astype(np.uint8)
            carry.append(equalized_image)

        self.enhance_img_list = carry

    def super_res(self):

        # for i in range(self.amount_of_images):
        #     # img.imwrite(self.base_name + "_" + str(i + 1)+ "_ENHANCE.png", self.enhance_img_list[i])

        image_carry = fusion_matrix(self.enhance_img_list)

        # img.imwrite(self.base_name + "_FINAL_ENHANCE.png", image_carry)

        error_acumulator = np.sqrt(np.sum(np.power(np.subtract(self.high_res_img ,image_carry), 2)) / (self.x_dim*self.y_dim))
        
        print( round(error_acumulator, 4))

    def choiser(self):
        if self.method_type is 0:
            self.method_0()

        elif self.method_type is 1:
            self.method_1()

        elif self.method_type is 2:
            self.method_2()

        elif self.method_type is 3:
            self.method_3()

        self.super_res()


def main():

    # auxiliar_path = "./imagens_trab2/"

    input_data_list = [ x.replace('\r\n','') for x in sys.stdin]

    image_small = input_data_list[0]
    image_big = input_data_list[1]
    enhance_method =  int (input_data_list[2])
    enhance_param =  float (input_data_list[3])


    enhance_object = enhance_image(image_small, 4, image_big, 2, enhance_param, enhance_method)
    enhance_object.choiser()

if __name__ == "__main__":
    # start = tm.perf_counter()
    main()
    # end = tm.perf_counter()
    # print (round(end - start, 5))