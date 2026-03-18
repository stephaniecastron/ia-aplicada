import "https://cdn.jsdelivr.net/npm/@tensorflow/tfjs@4.22.0/dist/tf.min.js";
import { workerEvents } from "../events/constants.js";

console.log("Model training worker initialized");
let _globalCtx = {};

const WEIGHTS = {
    price: 0.2,
    age: 0.1,
    color: 0.3,
    category: 0.4,
};

const normalize = (value, min, max) => (value - min) / (max - min) || 1;

async function trainModel({ users }) {
    console.log("Training model with users:", users);

    postMessage({
        type: workerEvents.progressUpdate,
        progress: { progress: 50 },
    });
    const catalog = await (await fetch("/data/products.json")).json();
    const context = makeContext(users, catalog);
    context.productVector = catalog.map((p) => {
        return {
            name: p.name,
            meta: { ...p },
            vector: encodeProduct(p, context).dataSync(),
        };
    });

	debugger;

    _globalCtx = context;
    console.log("Context for training:", context);

    debugger;
    postMessage({
        type: workerEvents.trainingLog,
        epoch: 1,
        loss: 1,
        accuracy: 1,
    });

    setTimeout(() => {
        postMessage({
            type: workerEvents.progressUpdate,
            progress: { progress: 100 },
        });
        postMessage({ type: workerEvents.trainingComplete });
    }, 1000);
}

function makeContext(users, catalog) {
    const ages = users.map((u) => u.age);
    const prices = catalog.map((p) => p.price);

    const minAge = Math.min(...ages);
    const maxAge = Math.max(...ages);

    const minPrice = Math.min(...prices);
    const maxPrice = Math.max(...prices);

    const colors = [...new Set(catalog.map((p) => p.color))];
    const categories = [...new Set(catalog.map((p) => p.category))];

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
        catalog.map((p) => {
            const avg = ageCounts[p.name]
                ? ageSums[p.name] / ageCounts[p.name]
                : midAge;

            return [p.name, normalize(avg, minAge, maxAge)];
        }),
    );

    return {
        catalog,
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

const oneHotWeight = (index, length, weight) =>
    tf.oneHot(index, length).cast("float32").mul(weight);

function encodeProduct(product, ctx) {
    //Normalizando dados para ficar de 0 a 1 e
    // aplicar o peso na recomendação
    const price = tf.tensor1d([
        normalize(
			product.price, 
			ctx.minPrice, 
			ctx.maxPrice
		) * WEIGHTS.price,
    ]);

    const age = tf.tensor1d([ctx.productsAvgAgeNorm[product.name] * WEIGHTS.age]) ?? 0.5

    const category = oneHotWeight(ctx.categoriesIndex[product.category], ctx.numCategories, WEIGHTS.category);

    const color = oneHotWeight(ctx.colorsIndex[product.color], ctx.numColors, WEIGHTS.color);

    return tf.concat([price, age, category, color]);
}

function recommend(user, ctx) {
    console.log("will recommend for user:", user);
    // postMessage({
    //     type: workerEvents.recommend,
    //     user,
    //     recommendations: []
    // });
}

const handlers = {
    [workerEvents.trainModel]: trainModel,
    [workerEvents.recommend]: (d) => recommend(d.user, _globalCtx),
};

self.onmessage = (e) => {
    const { action, ...data } = e.data;
    if (handlers[action]) handlers[action](data);
};
