package ai.vespa.schemals;

import ai.vespa.schemals.common.FileUtils;
import ai.vespa.schemals.index.SchemaIndex;
import ai.vespa.schemals.context.ParseContext;
import ai.vespa.schemals.schemadocument.SchemaDocument;
import ai.vespa.schemals.schemadocument.SchemaDocumentScheduler;
import ai.vespa.schemals.schemadocument.SchemaDocument.ParseResult;

import static com.yahoo.config.model.test.TestUtil.joinLines;
import com.yahoo.io.IOUtils;

import java.io.File;
import java.io.PrintStream;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.stream.Stream;

import org.eclipse.lsp4j.Diagnostic;
import org.eclipse.lsp4j.DiagnosticSeverity;
import org.eclipse.lsp4j.Position;
import org.junit.jupiter.api.DynamicTest;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.TestFactory;

import static org.junit.jupiter.api.Assertions.*;

public class SchemaParserTest {
    static long countErrors(List<Diagnostic> diagnostics) {
        return diagnostics.stream()
                          .filter(diag -> diag.getSeverity() == DiagnosticSeverity.Error || diag.getSeverity() == null)
                          .count();
    }

    static String constructDiagnosticMessage(List<Diagnostic> diagnostics, int indent) {
        String message = "";
        for (var diagnostic : diagnostics) {
            String severityString = "";
            if (diagnostic.getSeverity() == null) severityString = "No Severity";
            else severityString = diagnostic.getSeverity().toString();

            Position start = diagnostic.getRange().getStart();
            message += "\n" + new String(new char[indent]).replace('\0', '\t') +
                "Diagnostic" + "[" + severityString + "]: \"" + diagnostic.getMessage() + "\", at position: (" + start.getLine() + ", " + start.getCharacter() + ")";

        }
        return message;
    }

    ParseResult parseString(String input, String fileName) throws Exception {
        PrintStream logger = System.out;
        SchemaIndex schemaIndex = new SchemaIndex(logger);
        TestSchemaDiagnosticsHandler diagnosticsHandler = new TestSchemaDiagnosticsHandler(logger, new ArrayList<>());
        SchemaDocumentScheduler scheduler = new SchemaDocumentScheduler(logger, diagnosticsHandler, schemaIndex);
        schemaIndex.clearDocument(fileName);
        ParseContext context = new ParseContext(input, logger, fileName, schemaIndex, scheduler);
        context.useDocumentIdentifiers();
        return SchemaDocument.parseContent(context);
    }

    ParseResult parseString(String input) throws Exception {
        return parseString(input, "<FROMSTRING>");
    }

    ParseResult parseFile(String fileName) throws Exception {
        File file = new File(fileName);
        return parseString(IOUtils.readFile(file), fileName);
    }

    void checkFileParses(String fileName) throws Exception {
        var parseResult = parseFile(fileName);
        String testMessage = "For file: " + fileName + constructDiagnosticMessage(parseResult.diagnostics(), 1);
        assertEquals(0, countErrors(parseResult.diagnostics()), testMessage);
    }

    void checkFileFails(String fileName, int expectedErrors) throws Exception {
        var parseResult = parseFile(fileName);
        String testMessage = "For file: " + fileName + constructDiagnosticMessage(parseResult.diagnostics(), 1);
        assertEquals(expectedErrors, countErrors(parseResult.diagnostics()), testMessage);
    }

    void checkDirectoryParses(String directoryPath) throws Exception {
        String directoryURI = new File(directoryPath).toURI().toString();
        PrintStream logger = System.out;
        SchemaIndex schemaIndex = new SchemaIndex(logger);
        schemaIndex.setWorkspaceURI(directoryURI);
        List<Diagnostic> diagnostics = new ArrayList<>();
        SchemaDiagnosticsHandler diagnosticsHandler = new TestSchemaDiagnosticsHandler(logger, diagnostics);
        SchemaDocumentScheduler scheduler = new SchemaDocumentScheduler(logger, diagnosticsHandler, schemaIndex);
        List<String> schemaFiles = FileUtils.findSchemaFiles(directoryURI, logger);

        scheduler.setReparseDescendants(false);
        for (String schemaURI : schemaFiles) {
            scheduler.addDocument(schemaURI);
        }
        scheduler.reparseInInheritanceOrder();
        //scheduler.setReparseDescendants(true);

        diagnostics.clear();

        String testMessage = "\nFor directory: " + directoryPath;
        int numErrors = 0;
        for (String schemaURI : schemaFiles) {
            diagnostics.clear();
            var documentHandle = scheduler.getSchemaDocument(schemaURI);
            documentHandle.reparseContent();
            testMessage += "\n    File: " + schemaURI + constructDiagnosticMessage(diagnostics, 2);

            numErrors += countErrors(diagnostics);
        }

        assertEquals(0, numErrors, testMessage);
    }

    @Test
    void minimalSchemaParsed() throws Exception {
        String input = joinLines
            ("schema foo {",
             "  document foo {",
             "  }",
             "}");
        var parseResult = parseString(input);
        assertEquals(0, countErrors(parseResult.diagnostics()));
        assertTrue(parseResult.CST().isPresent(), "Parsing should return CST root!");
    }

    @TestFactory
    Stream<DynamicTest> generateGoodFileTests() {
        String[] filePaths = new String[] {
            "../../../config-model/src/test/derived/advanced/advanced.sd",
            "../../../config-model/src/test/derived/annotationsimplicitstruct/annotationsimplicitstruct.sd",
            "../../../config-model/src/test/derived/annotationsinheritance/annotationsinheritance.sd",
            "../../../config-model/src/test/derived/annotationsinheritance2/annotationsinheritance2.sd",
            "../../../config-model/src/test/derived/annotationsoutsideofdocument/annotationsoutsideofdocument.sd",
            "../../../config-model/src/test/derived/annotationspolymorphy/annotationspolymorphy.sd",
            "../../../config-model/src/test/derived/annotationsreference/annotationsreference.sd",
            "../../../config-model/src/test/derived/annotationsreference2/annotationsreference2.sd",
            "../../../config-model/src/test/derived/annotationssimple/annotationssimple.sd",
            "../../../config-model/src/test/derived/annotationsstruct/annotationsstruct.sd",
            "../../../config-model/src/test/derived/annotationsstructarray/annotationsstructarray.sd",
            "../../../config-model/src/test/derived/array_of_struct_attribute/test.sd",
            "../../../config-model/src/test/derived/arrays/arrays.sd",
            "../../../config-model/src/test/derived/attributeprefetch/attributeprefetch.sd",
             "../../../config-model/src/test/derived/attributerank/attributerank.sd",
            "../../../config-model/src/test/derived/attributes/attributes.sd",
            "../../../config-model/src/test/derived/combinedattributeandindexsearch/combinedattributeandindexsearch.sd",
            "../../../config-model/src/test/derived/complex/complex.sd",
            "../../../config-model/src/test/derived/emptydefault/emptydefault.sd",
            "../../../config-model/src/test/derived/exactmatch/exactmatch.sd",
            "../../../config-model/src/test/derived/fieldset/test.sd",
            "../../../config-model/src/test/derived/flickr/flickrphotos.sd",
            "../../../config-model/src/test/derived/function_arguments/test.sd",
            "../../../config-model/src/test/derived/function_arguments_with_expressions/test.sd",
            "../../../config-model/src/test/derived/gemini2/gemini.sd",
            "../../../config-model/src/test/derived/hnsw_index/test.sd",
            "../../../config-model/src/test/derived/id/id.sd",
            "../../../config-model/src/test/derived/indexinfo_fieldsets/indexinfo_fieldsets.sd",
            "../../../config-model/src/test/derived/indexinfo_lowercase/indexinfo_lowercase.sd",
            "../../../config-model/src/test/derived/indexschema/indexschema.sd",
            "../../../config-model/src/test/derived/indexswitches/indexswitches.sd",
            "../../../config-model/src/test/derived/integerattributetostringindex/integerattributetostringindex.sd",
            "../../../config-model/src/test/derived/language/language.sd",
            "../../../config-model/src/test/derived/lowercase/lowercase.sd",
            "../../../config-model/src/test/derived/mail/mail.sd",
            "../../../config-model/src/test/derived/map_attribute/test.sd",
            "../../../config-model/src/test/derived/map_of_struct_attribute/test.sd",
            "../../../config-model/src/test/derived/mlr/mlr.sd",
            "../../../config-model/src/test/derived/music/music.sd",
            "../../../config-model/src/test/derived/music3/music3.sd",
            "../../../config-model/src/test/derived/nearestneighbor/test.sd",
            "../../../config-model/src/test/derived/newrank/newrank.sd",
             "../../../config-model/src/test/derived/nuwa/newsindex.sd",
            "../../../config-model/src/test/derived/orderilscripts/orderilscripts.sd",
            "../../../config-model/src/test/derived/position_array/position_array.sd",
            "../../../config-model/src/test/derived/position_attribute/position_attribute.sd",
            "../../../config-model/src/test/derived/position_extra/position_extra.sd",
            "../../../config-model/src/test/derived/position_nosummary/position_nosummary.sd",
            "../../../config-model/src/test/derived/position_summary/position_summary.sd",
            "../../../config-model/src/test/derived/predicate_attribute/predicate_attribute.sd",
            "../../../config-model/src/test/derived/prefixexactattribute/prefixexactattribute.sd",
            "../../../config-model/src/test/derived/rankingexpression/rankexpression.sd",
            // "../../../config-model/src/test/derived/rankprofilemodularity/test.sd",
            "../../../config-model/src/test/derived/rankprofiles/rankprofiles.sd",
            "../../../config-model/src/test/derived/rankproperties/rankproperties.sd",
            "../../../config-model/src/test/derived/ranktypes/ranktypes.sd",
            "../../../config-model/src/test/derived/reference_fields/ad.sd",
            "../../../config-model/src/test/derived/reference_fields/campaign.sd",
            "../../../config-model/src/test/derived/renamedfeatures/foo.sd",
            "../../../config-model/src/test/derived/reserved_position/reserved_position.sd",
            "../../../config-model/src/test/derived/slice/test.sd",
            "../../../config-model/src/test/derived/streamingjuniper/streamingjuniper.sd",
            "../../../config-model/src/test/derived/streamingstruct/streamingstruct.sd",
            "../../../config-model/src/test/derived/streamingstructdefault/streamingstructdefault.sd",
            "../../../config-model/src/test/derived/structandfieldset/test.sd",
            "../../../config-model/src/test/derived/structanyorder/structanyorder.sd",
            "../../../config-model/src/test/derived/structinheritance/simple.sd",
            "../../../config-model/src/test/derived/tensor/tensor.sd",
            "../../../config-model/src/test/derived/tokenization/tokenization.sd",
            "../../../config-model/src/test/derived/types/types.sd",
            "../../../config-model/src/test/derived/uri_array/uri_array.sd",
            "../../../config-model/src/test/derived/uri_wset/uri_wset.sd",
            "../../../config-model/src/test/configmodel/types/types.sd",

            /*
             * CUSTOM TESTS
             * */
            "src/test/sdfiles/single/structinfieldset.sd"
        };

        return Arrays.stream(filePaths)
                     .map(path -> DynamicTest.dynamicTest(path, () -> checkFileParses(path)));
    }

    @TestFactory
    Stream<DynamicTest> generateGoodDirectoryTests() {
        String[] directories = new String[] {
            "../../../config-model/src/test/cfg/search/data/travel/schemas/",
            //"../../../config-model/src/test/configmodel/types/",
            "../../../config-model/src/test/derived/deriver/",
            "../../../config-model/src/test/derived/emptychild/",
            "../../../config-model/src/test/derived/imported_fields_inherited_reference/",
            "../../../config-model/src/test/derived/imported_position_field/",
            "../../../config-model/src/test/derived/imported_position_field_summary/",
            "../../../config-model/src/test/derived/imported_struct_fields/",
            //"../../../config-model/src/test/derived/importedfields/",
            "../../../config-model/src/test/derived/inheritance/",
            "../../../config-model/src/test/derived/inheritdiamond/",
            "../../../config-model/src/test/derived/inheritfromgrandparent/",
            "../../../config-model/src/test/derived/inheritfromparent/",
            "../../../config-model/src/test/derived/inheritstruct/",
            "../../../config-model/src/test/derived/namecollision/",
            "../../../config-model/src/test/derived/rankprofileinheritance/",
            //"../../../config-model/src/test/derived/schemainheritance/",
            "../../../config-model/src/test/derived/tensor2/",
            "../../../config-model/src/test/derived/twostreamingstructs/",
            //"../../../config-model/src/test/examples/",
        };

        return Arrays.stream(directories)
                     .map(dir -> DynamicTest.dynamicTest(dir, () -> checkDirectoryParses(dir)));
    }

    record BadFileTestCase(String filePath, int expectedErrors) {}

    @TestFactory
    Stream<DynamicTest> generateBadFileTests() {
        BadFileTestCase[] tests = new BadFileTestCase[] {
            new BadFileTestCase("../../../config-model/src/test/derived/inheritfromnull/inheritfromnull.sd", 1),
            new BadFileTestCase("../../../config-model/src/test/derived/structinheritance/bad.sd", 1), // TODO: check that the error is correct
            //new BadFileTestCase("src/test/sdfiles/single/rankprofilefuncs.sd", 2)
        };

        return Arrays.stream(tests)
                     .map(testCase -> DynamicTest.dynamicTest(testCase.filePath(), () -> checkFileFails(testCase.filePath(), testCase.expectedErrors())));
    }

}
