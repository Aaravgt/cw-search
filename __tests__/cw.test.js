"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
const cw_1 = require("../src/cw");
describe('CW validation examples', () => {
    test('CW(24,9) example passes', () => {
        const seq = [0, 0, 0, -1, -1, 0, 0, 0, 0, 0, 1, -1, 0, 0, 0, -1, 1, 0, 0, 1, 0, 0, -1, -1];
        const result = (0, cw_1.validateCW)(seq, 3);
        expect(result.valid).toBe(true);
        expect(result.weight).toBe(9);
        expect(Math.abs(result.sum)).toBe(3);
        expect(result.correlations.every(c => c === 0)).toBe(true);
    });
    test('CW(28,4) example passes', () => {
        const seq = [-1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0];
        const result = (0, cw_1.validateCW)(seq, 2);
        expect(result.valid).toBe(true);
        expect(result.weight).toBe(4);
        expect(Math.abs(result.sum)).toBe(2);
        expect(result.correlations.every(c => c === 0)).toBe(true);
    });
});
