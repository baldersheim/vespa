// Copyright Vespa.ai. Licensed under the terms of the Apache 2.0 license. See LICENSE in the project root.
package com.yahoo.language.opennlp;

import com.huaban.analysis.jieba.JiebaSegmenter;
import com.huaban.analysis.jieba.SegToken;
import com.yahoo.language.Language;
import com.yahoo.language.LinguisticsCase;
import com.yahoo.language.process.Normalizer;
import com.yahoo.language.process.SpecialTokenRegistry;
import com.yahoo.language.process.StemMode;
import com.yahoo.language.process.Token;
import com.yahoo.language.process.TokenType;
import com.yahoo.language.process.Tokenizer;
import com.yahoo.language.process.Transformer;
import com.yahoo.language.simple.SimpleNormalizer;
import com.yahoo.language.simple.SimpleToken;
import com.yahoo.language.simple.SimpleTokenType;
import com.yahoo.language.simple.SimpleTokenizer;
import com.yahoo.language.simple.SimpleTransformer;
import opennlp.tools.stemmer.Stemmer;
import opennlp.tools.stemmer.snowball.SnowballStemmer;

import java.util.ArrayList;
import java.util.List;
import java.util.Optional;

/**
 * Tokenizer using OpenNlp
 *
 * @author matskin
 * @author bratseth
 */
public class OpenNlpTokenizer implements Tokenizer {

    enum Mode { index, query }

    private final Mode mode;

    /** Whether to index cjk grams */
    private boolean createCjkGrams;

    private final Normalizer normalizer;
    private final Transformer transformer;

    /** Chinese segmenter if we're segmenting Chinese */
    private final Optional<JiebaSegmenter> chineseSegmenter;
    private final SimpleTokenizer simpleTokenizer;
    private final SpecialTokenRegistry specialTokenRegistry;

    public OpenNlpTokenizer() {
        this(new SimpleNormalizer(), new SimpleTransformer());
    }

    public OpenNlpTokenizer(Normalizer normalizer, Transformer transformer) {
        this(Mode.query, normalizer, transformer, false, false);
    }

    public OpenNlpTokenizer(Mode mode, Normalizer normalizer, Transformer transformer,
                            boolean cjk, boolean createCjkGrams) {
        this(mode, normalizer, transformer, cjk, createCjkGrams, new SpecialTokenRegistry(List.of()));
    }

    public OpenNlpTokenizer(Mode mode,
                            Normalizer normalizer,
                            Transformer transformer,
                            boolean cjk,
                            boolean createCjkGrams,
                            SpecialTokenRegistry specialTokenRegistry) {
        this.mode = mode;
        this.normalizer = normalizer;
        this.transformer = transformer;
        this.chineseSegmenter = cjk ? Optional.of(new JiebaSegmenter()) : Optional.empty();
        this.createCjkGrams = createCjkGrams;
        this.specialTokenRegistry = specialTokenRegistry;
        this.simpleTokenizer = new SimpleTokenizer(normalizer, transformer, specialTokenRegistry);
    }

    public OpenNlpTokenizer(Mode mode,
                            Normalizer normalizer,
                            Transformer transformer,
                            Optional<JiebaSegmenter> jiebaSegmenter,
                            boolean createCjkGrams,
                            SpecialTokenRegistry specialTokenRegistry) {
        this.mode = mode;
        this.normalizer = normalizer;
        this.transformer = transformer;
        this.chineseSegmenter = jiebaSegmenter;
        this.createCjkGrams = createCjkGrams;
        this.specialTokenRegistry = specialTokenRegistry;
        this.simpleTokenizer = new SimpleTokenizer(normalizer, transformer, specialTokenRegistry);
    }

    @Override
    public Iterable<Token> tokenize(String input, Language language, StemMode stemMode, boolean removeAccents) {
        if (chineseSegmenter.isPresent() && ( language == Language.CHINESE_SIMPLIFIED || language == Language.CHINESE_TRADITIONAL))
            return segmentChinese(input);

        Stemmer stemmer = stemmerFor(language, stemMode);
        if (stemmer == null)
            return simpleTokenizer.tokenize(input, language, stemMode, removeAccents);
        else
            return simpleTokenizer.tokenize(input, token -> processToken(token, language, stemMode, removeAccents, stemmer));
    }

    private Iterable<Token> segmentChinese(String input) {
        if (input.isEmpty()) return List.of();

        List<Token> tokens = new ArrayList<>();
        // In "search" mode Jieba will index grams of tokens longer than 2 characters
        var jiebaMode = ( mode == Mode.index && createCjkGrams ) ? JiebaSegmenter.SegMode.INDEX : JiebaSegmenter.SegMode.SEARCH;
        for (SegToken token : chineseSegmenter.get().process(input, jiebaMode)) {
            int nextCode = token.word.codePointAt(0);
            TokenType tokenType = SimpleTokenType.valueOf(nextCode);
            String originToken = input.substring(token.startOffset, token.startOffset + token.word.length());
            SimpleToken simpleToken = new SimpleToken(originToken)
                                              .setOffset(token.startOffset)
                                              .setType(tokenType)
                                              .setTokenString(token.word);
            tokens.add(simpleToken);
        }
        return tokens;
    }

    private String processToken(String token, Language language, StemMode stemMode, boolean removeAccents,
                                Stemmer stemmer) {
        token = normalizer.normalize(token);
        token = LinguisticsCase.toLowerCase(token);
        if (removeAccents)
            token = transformer.accentDrop(token, language);
        if (stemMode != StemMode.NONE)
            token = stemmer.stem(token).toString();
        return token;
    }

    private Stemmer stemmerFor(Language language, StemMode stemMode) {
        if (language == null || language == Language.ENGLISH || stemMode == StemMode.NONE) return null;
        SnowballStemmer.ALGORITHM algorithm = algorithmFor(language);
        if (algorithm == null) return null;
        return new SnowballStemmer(algorithm);
    }

    private SnowballStemmer.ALGORITHM algorithmFor(Language language) {
        return switch (language) {
            case DANISH -> SnowballStemmer.ALGORITHM.DANISH;
            case DUTCH -> SnowballStemmer.ALGORITHM.DUTCH;
            case FINNISH -> SnowballStemmer.ALGORITHM.FINNISH;
            case FRENCH -> SnowballStemmer.ALGORITHM.FRENCH;
            case GERMAN -> SnowballStemmer.ALGORITHM.GERMAN;
            case HUNGARIAN -> SnowballStemmer.ALGORITHM.HUNGARIAN;
            case IRISH -> SnowballStemmer.ALGORITHM.IRISH;
            case ITALIAN -> SnowballStemmer.ALGORITHM.ITALIAN;
            case NORWEGIAN_BOKMAL -> SnowballStemmer.ALGORITHM.NORWEGIAN;
            case NORWEGIAN_NYNORSK -> SnowballStemmer.ALGORITHM.NORWEGIAN;
            case PORTUGUESE -> SnowballStemmer.ALGORITHM.PORTUGUESE;
            case ROMANIAN -> SnowballStemmer.ALGORITHM.ROMANIAN;
            case RUSSIAN -> SnowballStemmer.ALGORITHM.RUSSIAN;
            case SPANISH -> SnowballStemmer.ALGORITHM.SPANISH;
            case SWEDISH -> SnowballStemmer.ALGORITHM.SWEDISH;
            case TURKISH -> SnowballStemmer.ALGORITHM.TURKISH;
            case ENGLISH -> SnowballStemmer.ALGORITHM.ENGLISH;
            default -> null;
        };
    }

    OpenNlpTokenizer withMode(Mode mode) {
        return new OpenNlpTokenizer(mode, normalizer, transformer, chineseSegmenter, createCjkGrams, specialTokenRegistry);
    }

}
