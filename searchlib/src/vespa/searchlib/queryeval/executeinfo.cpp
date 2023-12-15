// Copyright Vespa.ai. Licensed under the terms of the Apache 2.0 license. See LICENSE in the project root.

#include "executeinfo.h"

using vespalib::Doom;
namespace search::queryeval {

const ExecuteInfo ExecuteInfo::TRUE(true, 1.0, Doom::armageddon(), vespalib::ThreadBundle::trivial(), true, true);
const ExecuteInfo ExecuteInfo::FALSE(false, 1.0, Doom::armageddon(), vespalib::ThreadBundle::trivial(), true, true);

ExecuteInfo::ExecuteInfo() noexcept
    : ExecuteInfo(false, 1.0, Doom::armageddon(), vespalib::ThreadBundle::trivial(), true, true)
{ }

ExecuteInfo
ExecuteInfo::createForTest(bool strict, double hitRate) noexcept {
    return createForTest(strict, hitRate, Doom::armageddon());
}

}
