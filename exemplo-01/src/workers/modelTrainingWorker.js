import 'https://cdn.jsdelivr.net/npm/@tensorflow/tfjs@4.22.0/dist/tf.min.js';
import { workerEvents } from '../events/constants.js';

console.log('Model training worker initialized');
let _globalCtx = {};
let _model = {};

const WEIGHTS = {
    price: 0.2,
    age: 0.1,
    color: 0.3,
    category: 0.4,
};

const normalize = (value, min, max) => (value - min) / (max - min) || 1;

function encodeUser(user, ctx) {
    if (user.purchases.length) {
        return tf
            .stack(user.purchases.map((p) => encodeProduct(p, ctx)))
            .mean(0)
            .reshape([1, ctx.dimensions]);
    }

    return tf
        .concat1d([
            tf.zeros([1]), // preço é ignorado,
            tf.tensor1d([normalize(user.age, ctx.minAge, ctx.maxAge) * WEIGHTS.age]),
            tf.zeros([ctx.numCategories]), // cateoria ignorada,
            tf.zeros([ctx.numColors]), // color ignorada
        ])
        .reshape([1, ctx.dimensions]);
}

function createTrainingData(context) {
    const inputs = [];
    const labels = [];
    context.users
        .filter((u) => u.purchases.length)
        .forEach((u) => {
            const userVector = encodeUser(u, context).dataSync();
            context.products.forEach((product) => {
                const productVector = encodeProduct(product, context).dataSync();
                const label = u.purchases.some((purchase) => purchase.name === product.name) ? 1 : 0;
                // combinar usuario + products
                inputs.push([...userVector, ...productVector]);
                labels.push(label);
            });
        });

    return {
        xs: tf.tensor2d(inputs),
        ys: tf.tensor2d(labels, [labels.length, 1]),
        inputDimensions: context.dimensions * 2,
        //tamanho userVector + productVector
    };
}

// ====================================================================
// 📌 Exemplo de como um usuário é ANTES da codificação
// ====================================================================
/*
const exampleUser = {
    id: 201,
    name: 'Rafael Souza',
    age: 27,
    purchases: [
        { id: 8, name: 'Boné Estiloso', category: 'acessórios', price: 39.99, color: 'preto' },
        { id: 9, name: 'Mochila Executiva', category: 'acessórios', price: 159.99, color: 'cinza' }
    ]
};
*/

// ====================================================================
// 📌 Após a codificação, o modelo NÃO vê nomes ou palavras.
// Ele vê um VETOR NUMÉRICO (todos normalizados entre 0–1).
// Exemplo: [preço_normalizado, idade_normalizada, cat_one_hot..., cor_one_hot...]
//
// Suponha categorias = ['acessórios', 'eletrônicos', 'vestuário']
// Suponha cores      = ['preto', 'cinza', 'azul']
//
// Para Rafael (idade 27, categoria: acessórios, cores: preto/cinza),
// o vetor poderia ficar assim:
//
// [
//   0.45,            // peso do preço normalizado
//   0.60,            // idade normalizada
//   1, 0, 0,         // one-hot de categoria (acessórios = ativo)
//   1, 0, 0          // one-hot de cores (preto e cinza ativos, azul inativo)
// ]
//
// São esses números que vão para a rede neural.
// ====================================================================

// ====================================================================
// 🧠 Configuração e treinamento da rede neural
// ====================================================================

async function configureNeuralNetAndTrain(trainingData) {
    const model = tf.sequential();
    // Camada de entrada
    // - inputShape: Número de features por exemplo de treino (trainData.inputDim)
    //   Exemplo: Se o vetor produto + usuário = 20 números, então inputDim = 20
    // - units: 128 neurônios (muitos "olhos" para detectar padrões)
    // - activation: 'relu' (mantém apenas sinais positivos, ajuda a aprender padrões não-lineares)
    model.add(
        tf.layers.dense({
            inputShape: [trainingData.inputDimensions],
            units: 128,
            activation: 'relu',
        }),
    );
    // Camada oculta 1
    // - 64 neurônios (menos que a primeira camada: começa a comprimir informação)
    // - activation: 'relu' (ainda extraindo combinações relevantes de features)
    model.add(
        tf.layers.dense({
            units: 64,
            activation: 'relu', // Camada oculta adicional para aprender padrões mais complexos
        }),
    );
    // Camada oculta 2
    // - 32 neurônios (mais estreita de novo, destilando as informações mais importantes)
    //   Exemplo: De muitos sinais, mantém apenas os padrões mais fortes
    // - activation: 'relu'
    model.add(
        tf.layers.dense({
            units: 32,
            activation: 'relu',
        }),
    );
    // Camada de saída
    // - 1 neurônio porque vamos retornar apenas uma pontuação de recomendação
    // - activation: 'sigmoid' comprime o resultado para o intervalo 0–1
    //   Exemplo: 0.9 = recomendação forte, 0.1 = recomendação fraca
    model.add(
        tf.layers.dense({
            units: 1,
            activation: 'sigmoid',
        }),
    );

    model.compile({
        optimizer: tf.train.adam(0.001), // Otimizador Adam (ajusta os pesos de forma inteligente)
        loss: 'binaryCrossentropy', // Função de perda para classificação binária
        metrics: ['accuracy'], // Métrica para avaliar o desempenho durante o treino
    });

    await model.fit(trainingData.xs, trainingData.ys, {
        epochs: 100,
        batchSize: 32,
        shufle: true,
        callbacks: {
            onEpochEnd: (epoch, logs) => {
                postMessage({
                    type: workerEvents.trainingLog,
                    epoch: epoch,
                    loss: logs.loss,
                    accuracy: logs.acc,
                });
            },
        },
    });

    return model;
}

async function trainModel({ users }) {
    console.log('Training model with users:', users);

    postMessage({
        type: workerEvents.progressUpdate,
        progress: { progress: 50 },
    });
    const products = await (await fetch('/data/products.json')).json();
    const context = makeContext(users, products);
    context.productVector = products.map((p) => {
        return {
            name: p.name,
            meta: { ...p },
            vector: encodeProduct(p, context).dataSync(),
        };
    });

    _globalCtx = context;

    const trainingData = createTrainingData(context);
    _model = await configureNeuralNetAndTrain(trainingData);

    postMessage({
        type: workerEvents.progressUpdate,
        progress: { progress: 100 },
    });
    postMessage({ type: workerEvents.trainingComplete });
}

function makeContext(users, products) {
    const ages = users.map((u) => u.age);
    const prices = products.map((p) => p.price);

    const minAge = Math.min(...ages);
    const maxAge = Math.max(...ages);

    const minPrice = Math.min(...prices);
    const maxPrice = Math.max(...prices);

    const colors = [...new Set(products.map((p) => p.color))];
    const categories = [...new Set(products.map((p) => p.category))];

    const colorsIndex = Object.fromEntries(colors.map((c, i) => [c, i]));
    const categoriesIndex = Object.fromEntries(categories.map((c, i) => [c, i]));

    //Computar a média de idade dos comprados por produto
    // (ajuda a personalizr)
    const midAge = (minAge + maxAge) / 2;
    const ageSums = {};
    const ageCounts = {};
    users.forEach((u) => {
        u.purchases.forEach((p) => {
            ageSums[p.name] = (ageSums[p.name] || 0) + u.age;
            ageCounts[p.name] = (ageCounts[p.name] || 0) + 1;
        });
    });

    const productsAvgAgeNorm = Object.fromEntries(
        products.map((p) => {
            const avg = ageCounts[p.name] ? ageSums[p.name] / ageCounts[p.name] : midAge;

            return [p.name, normalize(avg, minAge, maxAge)];
        }),
    );

    return {
        products,
        users,
        colorsIndex,
        categoriesIndex,
        productsAvgAgeNorm,
        minPrice,
        maxPrice,
        numCategories: categories.length,
        numColors: colors.length,
        // price + age + color e categories
        dimensions: 2 + categories.length + colors.length,
    };
}

const oneHotWeight = (index, length, weight) => tf.oneHot(index, length).cast('float32').mul(weight);

function encodeProduct(product, ctx) {
    //Normalizando dados para ficar de 0 a 1 e
    // aplicar o peso na recomendação
    const price = tf.tensor1d([normalize(product.price, ctx.minPrice, ctx.maxPrice) * WEIGHTS.price]);

    const age = tf.tensor1d([ctx.productsAvgAgeNorm[product.name] * WEIGHTS.age]) ?? 0.5;

    const category = oneHotWeight(ctx.categoriesIndex[product.category], ctx.numCategories, WEIGHTS.category);

    const color = oneHotWeight(ctx.colorsIndex[product.color], ctx.numColors, WEIGHTS.color);

    return tf.concat([price, age, category, color]);
}

function recommend(user, context) {
    if (!_model) {
        return;
    }

    // 1️⃣ Converta o usuário fornecido no vetor de features codificadas
    //    (preço ignorado, idade normalizada, categorias ignoradas)
    //    Isso transforma as informações do usuário no mesmo formato numérico
    //    que foi usado para treinar o modelo.
    const userVector = encodeUser(user, context).dataSync();
    // Em aplicações reais:
    //  Armazene todos os vetores de produtos em um banco de dados vetorial (como Postgres, Neo4j ou Pinecone)
    //  Consulta: Encontre os 200 produtos mais próximos do vetor do usuário
    //  Execute _model.predict() apenas nesses produtos
    // 2️⃣ Crie pares de entrada: para cada produto, concatene o vetor do usuário
    //    com o vetor codificado do produto.
    //    Por quê? O modelo prevê o "score de compatibilidade" para cada par (usuário, produto).
    const inputs = context.productVector.map(({ vector }) => {
        return [...userVector, ...vector];
    });
    // 3️⃣ Converta todos esses pares (usuário, produto) em um único Tensor.
    //    Formato: [numProdutos, inputDim]
    const inputTensor = tf.tensor2d(inputs);
    // 4️⃣ Rode a rede neural treinada em todos os pares (usuário, produto) de uma vez.
    //    O resultado é uma pontuação para cada produto entre 0 e 1.
    //    Quanto maior, maior a probabilidade do usuário querer aquele produto.
    const predictions = _model.predict(inputTensor);
    // 5️⃣ Extraia as pontuações para um array JS normal.
    const scores = predictions.dataSync();
    const recommendations = context.productVector.map((item, index) => {
        return {
            ...item.meta,
            name: item.name,
            score: scores[index], // previsão do modelo para este produto
        };
    });
    const sortedItems = recommendations.sort((a, b) => b.score - a.score);
    // 8️⃣ Envie a lista ordenada de produtos recomendados
    //    para a thread principal (a UI pode exibi-los agora).
    postMessage({
        type: workerEvents.recommend,
        user,
        recommendations: sortedItems,
    });
}

const handlers = {
    [workerEvents.trainModel]: trainModel,
    [workerEvents.recommend]: (d) => recommend(d.user, _globalCtx),
};

self.onmessage = (e) => {
    const { action, ...data } = e.data;
    if (handlers[action]) handlers[action](data);
};
