package ai.vespa.schemals.completion;

import java.util.ArrayList;

import org.eclipse.lsp4j.CompletionItem;

import ai.vespa.schemals.completion.provider.*;
import ai.vespa.schemals.context.EventPositionContext;

public class SchemaCompletion {

    private static CompletionProvider[] providers = {
        new BodyKeywordCompletionProvider(),
        new TypeCompletionProvider(),
        new FieldsCompletionProvider()
    };

    public static ArrayList<CompletionItem> getCompletionItems(EventPositionContext context) {
        ArrayList<CompletionItem> ret = new ArrayList<CompletionItem>();

        for (CompletionProvider provider : providers) {
            if (provider.match(context)) {
                context.logger.println("Match with: " + provider.getClass().toString());
                ret.addAll(provider.getCompletionItems(context));
            }
        }

        return ret;
    }
}
