# EIP_SESSION3


new_model = Sequential()

new_model.add(SeparableConv2D(filters=64, kernel_size=3, activation='relu', input_shape=(32,32,3))) #30X30X64 RF: 3
new_model.add(BatchNormalization())
new_model.add(Dropout(0.2))
new_model.add(SeparableConv2D(filters=128, kernel_size=3, activation='relu')) #28X28X128 RF:5
new_model.add(BatchNormalization())
new_model.add(Dropout(0.2))
new_model.add(SeparableConv2D(filters=128, kernel_size=3, activation='relu')) #26X26X128 RF:7
new_model.add(BatchNormalization())
new_model.add(Dropout(0.2))

new_model.add(MaxPooling2D(pool_size=(2, 2))) #13X13X128 RF: 8
new_model.add(BatchNormalization())

new_model.add(SeparableConv2D(filters=64, kernel_size=3, activation='relu')) #11X11X64 RF: 12
new_model.add(BatchNormalization())
new_model.add(Dropout(0.2))
new_model.add(SeparableConv2D(filters=256, kernel_size=3, activation='relu')) #9X9X256 RF:16
new_model.add(BatchNormalization())
new_model.add(Dropout(0.2))
new_model.add(SeparableConv2D(filters=128, kernel_size=3, activation='relu')) #7X7X128 RF: 20
new_model.add(BatchNormalization())
new_model.add(Dropout(0.2))

new_model.add(MaxPooling2D(pool_size=(2, 2))) #3x3X128 RF: 22
new_model.add(BatchNormalization())

new_model.add(SeparableConv2D(filters=10, kernel_size=3, activation='relu')) #1X1X10 RF: 30
new_model.add(GlobalAveragePooling2D())

new_model.add(Dense(num_classes, activation='softmax'))
new_model.compile(optimizer=Adam(lr=0.003), loss='categorical_crossentropy', metrics=['accuracy'])
