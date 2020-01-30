import XCTest
@testable import S5TF

final class S5TFTests: XCTestCase {
    func testDataLoader() {
        let myDataLoader = MyDataLoader()
        let myIterator = S5TFDataIterator(dataLoader: myDataLoader)
        for i in myIterator.batched(3) {
            print(i)
        }
    }
}
